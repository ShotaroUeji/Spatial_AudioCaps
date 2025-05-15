#!/usr/bin/env python3
# foa2stereo.py — FOA(WXYZ)→Stereo(LR) 変換スクリプト
#
# 使い方:
#   python foa2stereo.py --foa pairs/97151/foa.wav --out pairs/97151/stereo.wav
#
# 依存: pip install soundfile numpy

import argparse
import numpy as np
import soundfile as sf

def decode_foa_to_stereo(foa: np.ndarray) -> np.ndarray:
    """
    foa: shape=(T,4) の FOA (W,X,Y,Z) WAV データ
    戻り値: shape=(T,2) のステレオ (L,R)
    """
    W = foa[:,0]
    Y = foa[:,2]   # ← ここを Y に
    # シンプルステレオデコード
    L = W + Y
    R = W - Y
    stereo = np.stack([L, R], axis=1)
    # クリッピング防止
    peak = np.max(np.abs(stereo))
    if peak > 1.0:
        stereo /= peak
    return stereo

def main():
    parser = argparse.ArgumentParser(description='FOA (WXYZ) → Stereo (LR) 変換')
    parser.add_argument('--foa', required=True, help='入力 FOA WAV ファイルパス')
    parser.add_argument('--out', default='stereo.wav', help='出力ステレオ WAV ファイル名')
    args = parser.parse_args()

    # FOA 読み込み
    foa, fs = sf.read(args.foa)
    if foa.ndim != 2 or foa.shape[1] != 4:
        raise ValueError(f'期待している FOA は 4ch ですが、読み込まれたチャンネル数 = {foa.shape[1]}')

    # デコード
    stereo = decode_foa_to_stereo(foa)

    # 書き出し
    sf.write(args.out, stereo, fs)
    print(f'✅ ステレオファイルを出力しました: {args.out} ({fs} Hz, {stereo.shape[1]}ch)')

if __name__ == '__main__':
    main()
