

```markdown
# Spatial-AudioCaps  🌐🎧  
**AudioCaps**（ステレオ音源＋字幕）を **空間拡張 (FOA/WXYZ＋空間キャプション)** し、  
ELSA 論文と同じ統計分布をもつ **Spatial-AudioCaps** をワンコマンド生成するツール群です。

```
project_root/
├── AudioCaps_csv/       # 公式 AudioCaps: train.csv / val.csv / test.csv
├── audiocaps_mp3/       # 公式 mp3 音源 (train/ val/ test/)
├── spatial_ranges.yml   # 各 split のパラメータ範囲
├── room_pool_trainval.json  # ← 自動生成
├── room_pool_test.json      # ← 自動生成
├── spatial_audiocaps/       # ← 生成されたペア一式
│   ├── train/  (◯◯件)
│   ├── val/    (◯◯件)
│   └── test/   (◯◯件)
├── manifest_train.csv   # ← 各 split ごとのペア一覧
├── manifest_val.csv
├── manifest_test.csv
└── scripts/
    ├── gen_room_pool.py    # 部屋プール生成
    ├── SpatialAudio.py     # mono→mic4+FOA(+stereo) 生成
    ├── SpatialCaps.py      # GPT で空間キャプション生成
    ├── make_pairs.py       # 全パイプライン統括
    └── foa2stereo.py       # FOA→stereo 単体変換 (参考)
```

---

## 1. 依存環境

```bash
# Python 3.9+
pip install -r requirements.txt
```

`requirements.txt`

```
numpy
scipy
pyroomacoustics
soundfile
librosa
pyyaml
pandas
openai>=1.0.0
tqdm
```

* **GPU不要** – ほぼ CPU-FFT と I/O がボトルネックです。  
  並列数を上げる場合は `make_pairs.py --workers <CPU実コア数>` にしてください。
* OpenAI API Key は環境変数 **`OPENAI_API_KEY`** に設定してください。

---

## 2. ステップバイステップ

### 2-1. 部屋プールを生成  
ELSA 論文準拠：  
* train/val 用 **8,952** 部屋  
* test 用 **4,970** 部屋（train/val の部分集合、ただしソース位置は非重複）

```bash
python scripts/gen_room_pool.py
```
出力  
```
room_pool_trainval.json : 8952 rooms
room_pool_test.json     : 4970 rooms  (subset)
```

### 2-2. 各 split を空間拡張  
例は 8 並列。`--stereo-out` を付けると LR ステレオも保存。

```bash
python scripts/make_pairs.py --split train \
       --csv AudioCaps_csv/train.csv \
       --audio-dir audiocaps_mp3/train \
       --out-dir spatial_audiocaps \
       --workers 8

python scripts/make_pairs.py --split val \
       --csv AudioCaps_csv/val.csv \
       --audio-dir audiocaps_mp3/val \
       --out-dir spatial_audiocaps \
       --workers 8

python scripts/make_pairs.py --split test \
       --csv AudioCaps_csv/test.csv \
       --audio-dir audiocaps_mp3/test \
       --out-dir spatial_audiocaps \
       --workers 8 \
       --spatial-parallel 1000  \
       --audio-parallel   1000
```

| オプション               | 意味 | 例 |
|--------------------------|------|----|
| `--spatial-parallel N`   | **同じ音源**で RIR だけ変えたペアを N 組生成（空間パラレル） | `1000` |
| `--audio-parallel N`     | **同じ部屋 / RIR** で違う音源を割り当てたペアを N 組生成（音源パラレル） | `1000` |
| `--stereo-out`           | FOA から同時に `stereo.wav` (W±Y) も保存 | |

### 2-3. 生成結果

* `spatial_audiocaps/<split>/<id>/`
  * `mic4.wav` … 正四面体マイク原信号 (4 ch)
  * `foa.wav`  … FOA(WXYZ) (4 ch, SN3D ACN)
  * `stereo.wav` … 任意、簡易 LR (W±Y)
  * `meta.yml` … 位置・部屋情報・T30 など
  * `caption.txt` … GPT による空間キャプション
* `manifest_<split>.csv`
  * id / caption / ファイルパス / ペア情報 等を一覧化  
    （トレーニング・評価でこの CSV だけ読めば OK）

---

## 3. Tips

| ヒント | 説明 |
|--------|------|
| **生成長** | `SpatialAudio.trim_pad(min_sec=4.0)` を短くすると容量削減可。 |
| **ピーク正規化** | FOA・Stereo は `peak>0.99` で自動リミッタ。 |
| **再キャプションだけ** | `SpatialCaps.py` を単体で呼び、`meta.yml` ＋ 原文から再生成可能。 |
| **FOA→Stereo 単体変換** | `python scripts/foa2stereo.py --foa …/foa.wav --out …/stereo.wav` |

---

## 4. ライセンス / 引用

* AudioCaps / YouTube 音源は各データセット・動画のライセンスに従います。  
  本ツール生成物の RIR・メタ情報コードは MIT ライセンスです。
* 学術用途で本ツールを利用した場合は、元論文 **ELSA (NeurIPS 2024)** を引用してください。

> Happy spatializing! 📡🎙️🎧
```

---

🎉 これで **セットアップから拡張データ生成まで README 一つで完結** です。
