#!/usr/bin/env python3
"""
make_pairs.py
    - 元 CSV を読み取り
    - SpatialAudio.spatial_foa() で FOA + meta を生成
    - SpatialCaps.rewrite_spatial_caption() で空間キャプションを生成
    - manifest.csv を書き出す
usage:
    python make_pairs.py \
        --csv  val_min.csv \
        --audio-dir wav \
        --out-dir pairs \
        --manifest manifest.csv \
        --workers 4
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))  # ← scripts/ ディレクトリを追加
from pathlib import Path
import argparse, csv, traceback, concurrent.futures as futures

import pandas as pd
import yaml

# ① 既存スクリプトを import して関数を直呼び
from SpatialAudio import spatial_foa          # ← generate FOA
from SpatialCaps  import rewrite_spatial_caption, load_meta  # ← GPT caption

def process_row(row, audio_dir: Path, out_dir: Path):
    """1本の音声を処理するユーティリティ。  
       例外は caller 側で握りつぶして continue できる形で戻す。"""
    audio_id   = str(row["audiocap_id"])
    caption    = row["caption"]
    in_file    = audio_dir / f"{audio_id}.mp3"
    sample_out = out_dir / audio_id        # eg. pairs/107890/

    if not in_file.exists():
        return dict(id=audio_id, ok=False, reason="audio_missing")

    try:
        # 1) FOA など生成（mic4.wav, foa.wav, meta.yml）
        spatial_foa(in_file, sample_out)

        # 2) meta.yml 読み込み → 空間キャプション生成
        meta = load_meta(sample_out / "meta.yml")
        new_cap = rewrite_spatial_caption(caption, meta)

        # 3) 生成キャプションをテキストで保存
        (sample_out / "caption.txt").write_text(new_cap, encoding="utf-8")

        return dict(
            id       = audio_id,
            ok       = True,
            caption  = new_cap,
            mic4     = str((sample_out/'mic4.wav').resolve()),
            foa      = str((sample_out/'foa.wav').resolve()),
            meta     = str((sample_out/'meta.yml').resolve())
        )

    except Exception as e:
        traceback.print_exc()
        return dict(id=audio_id, ok=False, reason=str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="元 CSV (id, caption 列があること)")
    ap.add_argument("--audio-dir", required=True, help="元 mp3/wav のフォルダ")
    ap.add_argument("--out-dir", required=True, help="生成物を書き出すルート")
    ap.add_argument("--manifest", default="manifest.csv",
                    help="ペア一覧を書き出す CSV (append)")
    ap.add_argument("--workers", type=int, default=1,
                    help="並列実行ワーカー数 (CPU/GPUに応じて調整)")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    needed_cols = {"audiocap_id", "caption"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"CSV には {needed_cols} が必要です")

    results = []
    with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_row, row, audio_dir, out_dir)
                for _, row in df.iterrows()]
        for f in futures.as_completed(futs):
            results.append(f.result())

    # manifest 追記
    man_path = Path(args.manifest)
    new_rows = [r for r in results if r["ok"]]
    write_header = not man_path.exists()
    with man_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f,
            fieldnames=["id","caption","mic4","foa","meta"], extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(new_rows)

    # サマリ
    total = len(results)
    ok    = sum(r["ok"] for r in results)
    print(f"\n✅ Done. success {ok}/{total}   manifest → {man_path}")

if __name__ == "__main__":
    main()
