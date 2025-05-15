#!/usr/bin/env python3
# SpatialAudio.py  –  正四面体マイク 4-ch ＋ FOA(WXYZ) 4-ch を生成
#   ・入力: mono WAV または MP3
#   ・出力: mic4.wav foa.wav rir.npy meta.yml
#
#   pip install pyroomacoustics numpy scipy soundfile pyyaml librosa

import random, sys, yaml
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import librosa  # MP3 / 各種フォーマット読み込み用

# ──────────────────── 設定ロード ────────────────────
_cfg_path = Path(__file__).with_name('spatial_ranges.yml')
with open(_cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

AREA_MIN, AREA_MAX = cfg['AREA_MIN'], cfg['AREA_MAX']
AZ_MIN, AZ_MAX       = cfg['AZ_MIN'],   cfg['AZ_MAX']
EL_MIN, EL_MAX       = cfg['EL_MIN'],   cfg['EL_MAX']
DIST_MIN, DIST_MAX   = cfg['DIST_MIN'], cfg['DIST_MAX']
ASL_MIN, ASL_MAX     = cfg['ASL_MIN'],  cfg['ASL_MAX']
ABS_MIN, ABS_MAX     = cfg['ABS_MIN'],  cfg['ABS_MAX']
# T30 が必要なら
#T30_MIN_MS, T30_MAX_MS = cfg['T30_MIN_MS'], cfg['T30_MAX_MS']

# ───── 前処理ユーティリティ ─── 無音除去＆最低長保証 ─────
def trim_and_pad(wav: np.ndarray, fs: int, min_sec: float = 4.0, top_db: float = 20.0):
    # 無音トリミング
    wav_trim, _ = librosa.effects.trim(wav, top_db=top_db)
    # 最短長保証
    min_len = int(min_sec * fs)
    if len(wav_trim) < min_len:
        n_rep = int(np.ceil(min_len / len(wav_trim)))
        wav_trim = np.tile(wav_trim, n_rep)
    # 4秒以上はそのまま返す
    return wav_trim

# ──────────────────── ユーティリティ ────────────────────
db   = lambda x: 20 * np.log10(max(x, 1e-12))
undb = lambda d: 10 ** (d / 20)

def active_rms(x, fs, hop=0.02, thr=-50):
    hop = int(hop * fs)
    rms = [np.sqrt(np.mean(x[i:i+hop]**2)) for i in range(0, len(x) - hop, hop)]
    act = [r for r in rms if db(r) > thr]
    return np.sqrt(np.mean(act**2)) if act else np.sqrt(np.mean(x**2))

def convolve_mc(sig, rir_lst):
    outs = [fftconvolve(sig, r.ravel(), mode='full') for r in rir_lst]
    Tmax = max(len(o) for o in outs)
    outs = [np.pad(o, (0, Tmax - len(o))) for o in outs]
    return np.stack(outs)

# ──────────────────── 仮想部屋＋マイク ────────────────────
def random_room(fs: int):
    area = random.uniform(AREA_MIN, AREA_MAX)
    w = h = np.sqrt(area)
    H = random.uniform(2.5, 4.0)
    dims = [w, h, H]

    alpha = random.uniform(ABS_MIN, ABS_MAX)
    room = pra.ShoeBox(dims, fs=fs,
                       materials=pra.Material(alpha),
                       max_order=6)

    # 正四面体マイク (半径 5 cm)
    r = 0.05; v = r / np.sqrt(3)
    tet = np.array([[ v,  v,  v],
                    [ v, -v, -v],
                    [-v,  v, -v],
                    [-v, -v,  v]]).T
    ctr = np.array(dims) / 2
    room.add_microphone_array(
        pra.MicrophoneArray(ctr.reshape(3,1) + tet, fs)
    )
    return room, ctr, alpha

# ──────────────────── メイン処理 ─────────────────────────
def spatial_foa(infile: Path, out_dir: Path):
    # 1) 入力ファイル読み込み
    try:
        wav, fs = sf.read(str(infile))
    except:
        wav, fs = librosa.load(str(infile), sr=None, mono=True)
    if wav.ndim > 1:
        wav = wav.mean(1)
    print(f'Loaded {infile.name}: {len(wav)/fs:.2f}s @ {fs} Hz')

    # 前処理: 無音除去＋最低4秒保証
    wav = trim_and_pad(wav, fs, min_sec=4.0, top_db=30.0)
    print(f'  → after trim/pad: {len(wav)/fs:.2f}s')

    # 2) 部屋と音源配置
    room, ctr, alpha = random_room(fs)
    dist = random.uniform(DIST_MIN, DIST_MAX)
    az   = np.deg2rad(random.uniform(AZ_MIN, AZ_MAX))
    el   = np.deg2rad(random.uniform(EL_MIN, EL_MAX))
    src  = ctr + dist * np.array([
               np.cos(el)*np.cos(az),
               np.cos(el)*np.sin(az),
               np.sin(el)
           ])
    src = np.clip(src, 0.2, np.array(room.shoebox_dim) - 0.2)
    room.add_source(src.tolist(), signal=wav)

    # 3) 残響付加
    room.compute_rir()
    rir_lst = [np.asarray(room.rir[m][0]).ravel() for m in range(len(room.rir))]
    Rmax = max(len(r) for r in rir_lst)
    rir_pad = [np.pad(r, (0, Rmax - len(r))) for r in rir_lst]
    sig_tet = convolve_mc(wav, rir_lst)

    # 4) アクティブレベル調整
    target = random.uniform(ASL_MIN, ASL_MAX)
    sig_tet *= undb(target - db(active_rms(sig_tet, fs)))

    # 5) FOA 変換 (SN3D, ACN)
    p1, p2, p3, p4 = sig_tet
    W = (p1 + p2 + p3 + p4) / 2
    X = (p1 + p2 - p3 - p4) / 2
    Y = (p1 - p2 - p3 + p4) / 2
    Z = (p1 - p2 + p3 - p4) / 2
    sig_foa = np.stack([W,X,Y,Z])

    # 5bis) ピーク正規化（クリップ防止）
    peak = np.max(np.abs(sig_foa))
    if peak > 0.99:
        sig_foa *= (0.99 / peak)
        print(f'⚠️ clip prevention: peak {peak:.3f} → 0.99')

    # 6) メタデータ計算 (Sabine式)
    V = np.prod(room.shoebox_dim)
    S = 2*(room.shoebox_dim[0]*room.shoebox_dim[1] +
           room.shoebox_dim[0]*room.shoebox_dim[2] +
           room.shoebox_dim[1]*room.shoebox_dim[2])
    T60 = 0.161 * V / (S * alpha)
    full_T30_ms = round(float(T60*500), 1)

    # 7) 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir/'mic4.wav', sig_tet.T, fs)
    sf.write(out_dir/'foa.wav', sig_foa.T, fs)
    np.save(out_dir/'rir.npy', np.stack(rir_pad))

    meta = {
        'fs': fs,
        'room_dim': [round(float(x),3) for x in room.shoebox_dim],
        'room_floor_m2': round(float(room.shoebox_dim[0]*room.shoebox_dim[1]),2),
        'source_pos': [round(float(x),3) for x in src],
        'azimuth_deg': round(float(np.degrees(np.arctan2(src[1]-ctr[1], src[0]-ctr[0]))),2),
        'elevation_deg': round(float(np.degrees(np.arcsin((src[2]-ctr[2])/dist))),2),
        'source_distance_m': round(float(dist),3),
        'fullband_T30_ms': full_T30_ms,
        'mic': 'tetra r=0.05',
        'target_asl_dB': round(float(target),2),
        'foa_format': 'SN3D ACN (WXYZ)'
    }
    (out_dir/'meta.yml').write_text(yaml.dump(meta, sort_keys=False))
    print('✅ files saved to', out_dir.resolve())


# ──────────────────── CLI ────────────────────
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python SpatialAudio.py <input.wav|mp3> <out_dir>')
        sys.exit(1)
    spatial_foa(Path(sys.argv[1]), Path(sys.argv[2]))
