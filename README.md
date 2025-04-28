# 📦 Spatial Audio–Caption ペア生成パイプライン

モノラル音声（MP3/WAV）を  
1. **合成残響**＋**FOA(4ch)** に変換  
2. 生成された **メタデータ** から **空間キャプション** を GPT-4o で自動生成  
3. 最終的に「ID／FOA WAV／Spatial Caption」の manifest を作成  

というステップを自動化します。

---

## 🗂 ディレクトリ構成例

Spatial_Audio_Caps/ ├── val_min/ ← 元データディレクトリ │ ├── val_min.csv ← [audiocap_id, filename, caption] の CSV │ ├── 97151.mp3 ← id が 97151 の元音声ファイル │ └── …
├── scripts/ │ ├── make_pairs.py ← パイプライン制御スクリプト │ ├── SpatialAudio.py ← 部屋シミュ→FOA生成モジュール │ └── SpatialCaps.py ← GPT-4o キャプション生成モジュール ├── pairs/ ← 出力先ディレクトリ（実行時に自動生成） │ └── 97151/ ← 例：ID ごとにサブフォルダ │ ├── mic4.wav ← 4ch マイク信号 │ ├── foa.wav ← 4ch FOA (WXYZ) │ ├── rir.npy ← 各チャンネルの RIR を ndarray 保存 │ └── meta.yml ← 方位・距離・T30 などのメタ情報 ├── manifest.csv ← [(id, foa_path, spatial_caption), …] 一覧 └── requirements.txt ← 必要パッケージ一覧

yaml
コピーする
編集する

---

## 🔧 依存パッケージ

```text
pyroomacoustics
numpy
scipy
librosa
soundfile
pyyaml
pandas
openai>=1.3.5
tqdm
requirements.txt にまとめておけば以下で一括インストールできます:

bash
コピーする
編集する
pip install -r requirements.txt
🔑 OpenAI API キー設定
GPT-4o を呼び出す際に必要です:

powershell
コピーする
編集する
# PowerShell (Windows)
$Env:OPENAI_API_KEY = "sk-…"

# bash/macOS
export OPENAI_API_KEY="sk-…"
🚀 実行手順
準備

val_min/val_min.csv：3列 audiocap_id,filename,caption

val_min/<filename>.mp3|wav：CSV と同名の音声ファイルを配置

ペア生成

bash
コピーする
編集する
python scripts/make_pairs.py \
  --csv       val_min/val_min.csv \
  --audio-dir val_min/ \
  --out-dir   pairs/ \
  --manifest  manifest.csv
--csv : 入力 CSV

--audio-dir : モノラル音声ファイル群ディレクトリ

--out-dir : FOA＋meta を出力する先

--manifest : 書き出す manifest.csv

処理内容

各行の filename → scripts/SpatialAudio.py の spatial_foa() を呼び出し

入力：mono WAV/MP3

出力：mic4.wav, foa.wav, rir.npy, meta.yml

scripts/SpatialCaps.py の rewrite_spatial_caption() により

meta.yml ＋元 caption をプロンプト化 → GPT-4o で「空間キャプション」を生成

pairs/<id>/… にファイルを保存し、

manifest.csv に id, pairs/<id>/foa.wav, new_caption を追記

⚙️ 各スクリプト概要
scripts/SpatialAudio.py
spatial_foa(infile: Path, out_dir: Path)

モノラル音声読み込み

ランダム合成部屋生成 (床面積13.3～277.4m²、吸音率ランダム)

ランダム方位・距離・仰角でソース配置

RIR シミュレーション → マイク4ch 信号＋FOA(WXYZ) 生成

アクティブ音量を 85–100dB-ASL に自動調整

RIR, メタデータ (YAML), mic4.wav, foa.wav を出力

scripts/SpatialCaps.py
rewrite_spatial_caption(original: str, meta: dict) -> str

meta.yml から azimuth_deg, source_distance_m, room_floor_m2, fullband_T30_ms 取得

「far/near」「front/left/…」「small/large」「highly reverberant/…」にマッピング

プロンプトを組み立て → OpenAI Chat API (GPT-4o) 呼び出し

返ってきたリライト文（空間キャプション）を返却

scripts/make_pairs.py
CSV を逐次読み込み

各行ごとに上記 2 モジュールを連携

pairs/<id>/… に出力しつつ manifest.csv を更新

💡 注意・TIPS
API Key エラー → OPENAI_API_KEY の設定 or キー有効性を要確認

MP3 読み込み失敗 → ffmpeg インストール or librosa にフォールバック

並列化 → multiprocessing.Pool／tqdm.contrib.concurrent 等で高速化OK

モデル変更 → SpatialCaps.py の model='gpt-4o' 部分を gpt-3.5-turbo 等に