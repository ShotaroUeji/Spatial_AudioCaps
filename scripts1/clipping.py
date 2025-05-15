import soundfile as sf
import numpy as np

y, fs = sf.read('pairs/99850/foa.wav')  # (T,4)
peak = np.max(np.abs(y))
print(f'最大ピーク振幅: {peak:.3f}')
# → 1.0越え or 0.99ちかければクリップが起きています

# チャンネルごとにクリップ数
for ch in range(y.shape[1]):
    clip_count = np.sum(np.abs(y[:,ch]) >= 0.9999)
    total = y.shape[0]
    print(f'Ch{ch}: クリップ数 {clip_count} / {total} ({100*clip_count/total:.3f} %)')
