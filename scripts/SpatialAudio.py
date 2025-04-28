#!/usr/bin/env python3
# o3_foa.py  –  正四面体マイク 4-ch ＋ FOA(WXYZ) 4-ch を生成
#   ・入力:  mono WAV または MP3
#   ・出力:  mic4.wav  foa.wav  rir.npy  meta.yml
#
#   pip install pyroomacoustics numpy scipy soundfile pyyaml librosa

import random, sys, yaml
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import librosa                               # MP3 デコード用

# ───────────────── Spatial-LibriSpeech 範囲 ──────────────────
AREA_MIN, AREA_MAX = 13.3, 277.4            # 床面積 [m²]
AZ_MIN , AZ_MAX    = -180 , 180             # 方位角 [°]
EL_MIN , EL_MAX    = -47.5, 48.7            # 仰角   [°]
DIST_MIN, DIST_MAX = 0.5 , 4.0              # 距離   [m]
ASL_MIN , ASL_MAX  = 85  , 100              # dB-ASL
ABS_MIN , ABS_MAX  = 0.02, 0.90             # 吸音率 α の乱数範囲

# ───────────────── ユーティリティ ──────────────────────────
db   = lambda x: 20*np.log10(max(x,1e-12))
undb = lambda d: 10**(d/20)

def active_rms(x, fs, hop=0.02, thr=-50):
    hop = int(hop*fs)
    rms = [np.sqrt(np.mean(x[i:i+hop]**2)) for i in range(0,len(x)-hop,hop)]
    act = [r for r in rms if db(r) > thr]
    return np.sqrt(np.mean(act**2)) if act else np.sqrt(np.mean(x**2))

def convolve_mc(sig, rir_lst):
    outs = [fftconvolve(sig, r.ravel(), mode='full') for r in rir_lst]
    Tmax = max(len(o) for o in outs)
    outs = [np.pad(o,(0,Tmax-len(o))) for o in outs]
    return np.stack(outs)

# ───────────────── 仮想部屋＋マイク ─────────────────────────
def random_room(fs: int):
    area = random.uniform(AREA_MIN, AREA_MAX)
    w = h = np.sqrt(area)
    H = random.uniform(2.5, 4.0)
    dims = [w, h, H]

    alpha = random.uniform(ABS_MIN, ABS_MAX)          # 吸音率を乱数化
    room  = pra.ShoeBox(dims, fs=fs,
                        materials=pra.Material(alpha),
                        max_order=15)

    # 正四面体マイク (半径 5 cm)
    r = 0.05; v = r/np.sqrt(3)
    tet = np.array([[ v,  v,  v],
                    [ v, -v, -v],
                    [-v,  v, -v],
                    [-v, -v,  v]]).T
    ctr = np.array(dims)/2
    room.add_microphone_array(pra.MicrophoneArray(ctr.reshape(3,1)+tet, fs))
    return room, ctr, alpha

# ───────────────── メイン処理 ───────────────────────────────
def spatial_foa(infile: Path, out_dir: Path):
    # 1) 入力ファイル読み込み (MP3 は librosa)
    try:
        wav, fs = sf.read(str(infile))
    except Exception:        #sf.readが失敗したら、librosaでmp3/wavを読み込む
        wav, fs = librosa.load(str(infile), sr=None, mono=True)
    if wav.ndim > 1: wav = wav.mean(1)
    print(f'Loaded {infile.name}: {len(wav)/fs:.2f}s @ {fs} Hz')

    # 2) 部屋と音源
    room, ctr, alpha = random_room(fs)
    dist = random.uniform(DIST_MIN, DIST_MAX)
    az   = np.deg2rad(random.uniform(AZ_MIN, AZ_MAX))
    el   = np.deg2rad(random.uniform(EL_MIN, EL_MAX))
    src  = ctr + dist*np.array([np.cos(el)*np.cos(az),
                                np.cos(el)*np.sin(az),
                                np.sin(el)])
    src  = np.clip(src, 0.2, np.array(room.shoebox_dim)-0.2)
    room.add_source(src.tolist(), signal=wav)

    # 3) 残響付加
    room.compute_rir()
    rir_lst = [np.asarray(room.rir[m][0]).ravel() for m in range(len(room.rir))]
    Rmax = max(len(r) for r in rir_lst)
    rir_pad = [np.pad(r, (0, Rmax-len(r))) for r in rir_lst]
    sig_tet = convolve_mc(wav, rir_lst)      # (4,T)

    # 4) アクティブレベル調整
    target = random.uniform(ASL_MIN, ASL_MAX)
    sig_tet *= undb(target - db(active_rms(sig_tet, fs)))

    # 5) FOA 変換 (SN3D, ACN)
    p1,p2,p3,p4 = sig_tet
    W = (p1+p2+p3+p4)/2
    X = (p1+p2-p3-p4)/2
    Y = (p1-p2+p3-p4)/2
    Z = (p1-p2-p3+p4)/2
    sig_foa = np.stack([W,X,Y,Z])

    # 6) メタデータ計算 (Sabine 式で T30 推定)
    V = np.prod(room.shoebox_dim)                            # 体積
    S = 2*(room.shoebox_dim[0]*room.shoebox_dim[1] +
           room.shoebox_dim[0]*room.shoebox_dim[2] +
           room.shoebox_dim[1]*room.shoebox_dim[2])          # 表面積
    T60 = 0.161 * V / (S * alpha)
    full_T30_ms = round(float(T60*500), 1)                   # ms

    # 7) 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir/'mic4.wav', sig_tet.T, fs)
    sf.write(out_dir/'foa.wav',  sig_foa.T, fs)
    np.save(out_dir/'rir.npy', np.stack(rir_pad))            # RIR (可変長なのでパディング無し)

    meta = dict(
        fs = fs,
        room_dim        = [round(float(x),3) for x in room.shoebox_dim],
        room_floor_m2   = round(float(room.shoebox_dim[0]*room.shoebox_dim[1]),2),
        source_pos      = [round(float(x),3) for x in src],
        azimuth_deg     = round(float(np.degrees(np.arctan2(src[1]-ctr[1], src[0]-ctr[0]))),2),
        elevation_deg   = round(float(np.degrees(np.arcsin((src[2]-ctr[2])/dist))),2),
        source_distance_m = round(float(dist),3),
        fullband_T30_ms = full_T30_ms,
        mic = 'tetra r=0.05',
        target_asl_dB = round(float(target),2),
        foa_format = 'SN3D ACN (WXYZ)'
    )
    (out_dir/'meta.yml').write_text(yaml.dump(meta, sort_keys=False))
    print('✅ files saved to', out_dir.resolve())

# ───────────────── CLI ────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python o3_foa.py <input.wav|mp3> <out_dir>')
        sys.exit(1)
    spatial_foa(Path(sys.argv[1]), Path(sys.argv[2]))
