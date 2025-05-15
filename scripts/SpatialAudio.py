#!/usr/bin/env python3
"""SpatialAudio.py  –  mono/stereo 音声 → 4ch 正四面体 + FOA + (任意で stereo)
   ・RIR は gen_room_pool.py が作った JSON から 1 件取得
   ・split 指定で trainval / test どちらのプールを使うか自動切替
"""
import random, json, yaml, math, sys
from pathlib import Path
import numpy as np, librosa, soundfile as sf
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from typing import Literal
# ───────────────── 設定ロード
cfg = yaml.safe_load(Path("spatial_ranges.yml").read_text())

# ──────────────────── 量子化グリッド & ヘルパ ──────────────────────
# 距離: 1 cm (=0.01 m) 単位  /  角度: 1° 単位
GRID_CM  = 1       # [cm]
GRID_DEG = 1        # [deg]



POOL = {
    "train": json.loads(Path("room_pool_trainval.json").read_text()),
    "val"  : json.loads(Path("room_pool_trainval.json").read_text()),
    "test" : json.loads(Path("room_pool_test.json").read_text()),
}


def _snap(val: float, grid: float) -> float:
    """val を grid 単位に丸めて返す"""
    return round(val / grid) * grid


def _load_audio(infile: Path):
    """
    ① soundfile で読めれば最速（libsndfile が mp3 対応なら mp3 も OK）
    ② 失敗したら librosa+audioread にフォールバック
       mono=True で単 ch、 sr=None で原本サンプリング周波数
    """
    try:
        wav, fs = sf.read(str(infile))
        if wav.ndim > 1:           # ステレオ→mono 平均
            wav = wav.mean(axis=1)
    except Exception:              # mp3 コーデック無しなど
        wav, fs = librosa.load(str(infile), sr=None, mono=True)
    return wav.astype(np.float32), fs



def _rand_room(split):
    return random.choice(POOL[split])

# ──────────────────────── 位置サンプリング ─────────────────────────
def _rand_position(
    split    : str,
    ctr      : np.ndarray,
    dist_rng : tuple[float,float] | None = None,
    el_rng   : tuple[float,float] | None = None
) -> tuple[np.ndarray,float,float,float]:
    """
    • split ('train'|'val'|'test') から専用レンジを自動選択
    • 1 cm / 1° で量子化し
    • (dist, az, el) セル和の偶奇で
      train/val = 偶数セル, test = 奇数セル
      に振り分けて “絶対重複なし” を保証
    """

    # split ごとのレンジを取得
    if dist_rng is None or el_rng is None:
        rng_cfg = cfg["TEST"] if split == "test" else cfg["TRAINVAL"]
        dist_min, dist_max = rng_cfg["DIST_MIN"], rng_cfg["DIST_MAX"]
        el_min,   el_max   = rng_cfg["EL_MIN"],   rng_cfg["EL_MAX"]
    else:
        dist_min, dist_max = dist_rng
        el_min,   el_max   = el_rng

    while True:
        # 連続乱数 sampling
        dist   = random.uniform(dist_min, dist_max)   # [m]
        az_deg = random.uniform(-180.0, 180.0)        # [deg]
        el_deg = random.uniform(el_min, el_max)       # [deg]

        # 量子化 (1 cm / 1°)
        dist_q = _snap(dist, GRID_CM / 100)           # [m]
        az_q   = _snap(az_deg, GRID_DEG)
        el_q   = _snap(el_deg, GRID_DEG)

        # セル番号を整数化して parity 判定
        cell_d = int(round(dist_q / (GRID_CM / 100)))
        cell_a = int(round((az_q + 180.0) / GRID_DEG))
        cell_e = int(round((el_q - el_min) / GRID_DEG))
        parity = (cell_d + cell_a + cell_e) & 1        # 0=偶,1=奇
        want   = 1 if split == "test" else 0
        if parity != want:
            continue  # split に合わなければ再 sampling

        # 条件クリア。src 座標を計算して返す
        az_rad = math.radians(az_q)
        el_rad = math.radians(el_q)
        src = ctr + dist_q * np.array([
            math.cos(el_rad)*math.cos(az_rad),
            math.cos(el_rad)*math.sin(az_rad),
            math.sin(el_rad)
        ])
        return src, dist_q, az_q, el_q
    

def trim_pad(x, fs, min_sec=4.0):
    y, _ = librosa.effects.trim(x, top_db=30)
    need = int(min_sec*fs) - len(y)
    if need>0: y = np.tile(y, math.ceil(need/len(y)))[:need+len(y)]
    return y

def spatial_foa(in_wav: Path | str, out_dir: Path, split: Literal["train","val","test"],
                room_conf=None, stereo_out=False):
    in_wav = Path(in_wav)
    wav, fs = _load_audio(in_wav)

    wav = trim_pad(wav, fs)
    print(f"Loaded {in_wav.name} : {len(wav)/fs:.2f}s @ {fs} Hz")
    room_cfg = room_conf if room_conf else _rand_room(split)
    w,h,H = room_cfg["dims"]; alpha = room_cfg["alpha"]
    room = pra.ShoeBox([w,h,H], fs=fs,
                       materials=pra.Material(alpha), max_order=10)
    ctr = np.array([w/2,h/2,H/2])

    rng = cfg["TEST"] if split=="test" else cfg["TRAINVAL"]
    src, dist, az_deg, el_deg = _rand_position(
        split, ctr, (rng["DIST_MIN"], rng["DIST_MAX"]),
        (rng["EL_MIN"],   rng["EL_MAX"])
    )

    # ─── ソース位置を必ず部屋の内側に収める ──────────
    # 端から0.1mずつマージンを取ってクリップ
    dims = np.array(room.shoebox_dim)
    margin = 0.1
    src = np.clip(src, margin, dims - margin)
    room.add_source(src.tolist(), signal=wav)
    # 正四面体マイク
    r=0.05; v=r/math.sqrt(3)
    tet = np.array([[ v, v, v], [ v,-v,-v], [-v, v,-v], [-v,-v, v]]).T
    room.add_microphone_array(pra.MicrophoneArray(ctr.reshape(3,1)+tet,fs))
    room.compute_rir()
    # ---- RIR 畳み込み（4ch）-----------------------------------
    # room.rir[m][s]   ← m=マイク index, s=ソース index (今回 0 のみ)
    outs = [
        fftconvolve(
            wav,
            np.asarray(room.rir[m][0]).ravel(),   # ← マイク m の RIR
            mode="full"
        )
        for m in range(len(room.rir))             # 4 本ループ
    ]

    Tmax = max(len(o) for o in outs)              # 最長サンプル数
    outs = [np.pad(o, (0, Tmax - len(o))) for o in outs]  # 右側ゼロ詰め

    mic4 = np.stack(outs)                         # shape = (4, Tmax)

    # Level & Clip
    #peak=np.max(np.abs(mic4)); 
   # if peak>0.99: mic4 *= 0.99/peak

    W=(mic4[0]+mic4[1]+mic4[2]+mic4[3])/2
    X=(mic4[0]+mic4[1]-mic4[2]-mic4[3])/2
    Y=(mic4[0]-mic4[1]-mic4[2]+mic4[3])/2
    Z=(mic4[0]-mic4[1]+mic4[2]-mic4[3])/2
    foa=np.stack([W,X,Y,Z])

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir/'mic4.wav', mic4.T, fs)
    sf.write(out_dir/'foa.wav',  foa.T,  fs)
    if stereo_out:
        # ★ W±Y デコード  + クリップ抑制
        L = foa[0] + foa[2]          # W+Y
        R = foa[0] - foa[2]          # W−Y
        stereo = np.stack([L, R])
        #peak = np.max(np.abs(stereo))
        #if peak > 1.0:
         #   stereo /= peak
        sf.write(out_dir/'stereo.wav', stereo.T, fs)
    meta=dict(
        dims=room_cfg["dims"],
        area_m2=room_cfg["area_m2"],
        alpha=float(alpha),
        fullband_T30_ms=room_cfg["T30_ms"],
        source_distance_m=round(dist,3),
        azimuth_deg=round(az_deg,2),
        elevation_deg=round(el_deg,2),
        split=split
    )
    Path(out_dir/'meta.yml').write_text(yaml.dump(meta, sort_keys=False))

# CLI ≒ quick test
if __name__=="__main__":
    if len(sys.argv)!=4:
        print("usage: SpatialAudio.py in.wav out_dir train|val|test")
        sys.exit(1)
    spatial_foa(Path(sys.argv[1]), Path(sys.argv[2]), split=sys.argv[3],
                stereo_out=True)
