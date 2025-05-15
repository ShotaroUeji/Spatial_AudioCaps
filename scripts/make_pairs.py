#!/usr/bin/env python3
"""
make_pairs.py  –  AudioCaps を FOA 拡張 + GPT 空間キャプション化

例)
python scripts/make_pairs.py \
    --split test \
    --csv  val_min.csv \
    --audio-dir val_min \
    --out-dir aa \
    --workers 8 \
    --spatial-parallel 6
"""
from pathlib import Path
import random, argparse, yaml, csv, traceback
import concurrent.futures as fut
import pandas as pd

from SpatialAudio import spatial_foa
from SpatialCaps  import rewrite_spatial_caption
from json import loads

# ────────────────────────────────────────────────────────────
def gen_sample(row              : dict,
               split            : str,
               out_root         : Path,
               suffix:str       = "",
               pair_id:str      = "",
               pair_type:str    = "",
               room_conf:dict   = None,
               stereo:bool      = "") -> dict:
    """
    1 本の mp3 を
      • FOA(+mic4) 生成
      • meta.yml -> GPT キャプション生成
      • caption.txt 保存
    """
    in_wav = Path(row["file"])              # すでに絶対パスにしてある
    if not in_wav.exists():
        return {"ok": False, "id": row["id"], "reason": "audio_missing"}

    out_dir = out_root / f"{row['audiocap_id']}{suffix}"

    try:
        spatial_foa(in_wav, out_dir, split=split,
                    room_conf=room_conf, stereo_out=stereo)
        meta = yaml.safe_load((out_dir / "meta.yml").read_text())
        cap  = rewrite_spatial_caption(row["caption"], meta)
        (out_dir / "caption.txt").write_text(cap, encoding="utf8")

        return dict(
            ok=True,
            id=out_dir.name,
            pair_id=pair_id,
            pair_type=pair_type or "single",
            caption=cap,
            mic4=str((out_dir / "mic4.wav").resolve()),
            foa =str((out_dir / "foa.wav").resolve()),
            stereo=str((out_dir / "stereo.wav").resolve()),
            meta=str((out_dir / "meta.yml").resolve()),
        )

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "id": row["id"], "reason": str(e)}


# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spatial-parallel", type=int, default=0)
    ap.add_argument("--audio-parallel",   type=int, default=0)
    ap.add_argument("--stereo", default=False, type=bool)
    args = ap.parse_args()

    # ── 入力 CSV を読み込み & ファイルパス列を作成
    audio_dir = Path(args.audio_dir).resolve()
    df = pd.read_csv(args.csv)
    df["id"] = df["audiocap_id"]  
    df["file"] = df["audiocap_id"].astype(str).apply(
        lambda x: str(audio_dir / f"{x}.mp3"))
    rows = df.to_dict("records")
    random.shuffle(rows)

    # 出力ルート
    out_root = Path(args.out_dir) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    futures = []

    # --------------------------------------------------------


# ────────────────────────────────────────────────────────────
    with fut.ProcessPoolExecutor(args.workers) as ex:

        # ① 空間パラレル（同一音声・別 RIR）
        if args.spatial_parallel:
            # split に応じた room_pool を読み込む
            pool_file = "room_pool_trainval.json" if args.split in ("train", "val") else "room_pool_test.json"
            room_pool = loads(Path(pool_file).read_text())

            # pop できる行数に合わせて最大件数を clamp
            max_sp = min(args.spatial_parallel, len(rows))
            if max_sp < args.spatial_parallel:
                print(f"⚠ spatial_parallel too large ({args.spatial_parallel}), reducing to {max_sp}")

            for i in range(max_sp):
                pair_id = f"SP{str(i).zfill(4)}"
                # 必要なら行が残っているかチェック
                if not rows:
                    raise RuntimeError("Not enough rows remaining for spatial_parallel")
                r = rows.pop(0)
                for suf in ("A", "B"):
                    futures.append(
                        ex.submit(
                            gen_sample, r, args.split, out_root,
                            suffix=f"_{suf}",
                            pair_id=pair_id,
                            pair_type="spatial",
                            stereo=args.stereo
                        )
                    )

        # ② 音源パラレル（別音声・同一 RIR）
        if args.audio_parallel:
            # split に応じた room_pool を読み込む
            pool_file = "room_pool_trainval.json" if args.split in ("train", "val") else "room_pool_test.json"
            room_pool = loads(Path(pool_file).read_text())

            # pop できる行数に合わせて最大件数を clamp (各ジョブで2行消費)
            max_ap = min(args.audio_parallel, len(rows) // 2)
            if max_ap < args.audio_parallel:
                print(f"⚠ audio_parallel too large ({args.audio_parallel}), reducing to {max_ap}")

            for i in range(max_ap):
                pair_id = f"AP{str(i).zfill(4)}"
                room_conf = random.choice(room_pool)
                for _ in range(2):
                    if not rows:
                        raise RuntimeError("Not enough rows remaining for audio_parallel")
                    r = rows.pop(0)
                    futures.append(
                        ex.submit(
                            gen_sample, r, args.split, out_root,
                            pair_id=pair_id,
                            pair_type="audio",
                            room_conf=room_conf,
                            stereo=args.stereo
                        )
                    )

        # ③ 残り
        for r in rows:
            futures.append(
                ex.submit(
                    gen_sample, r, args.split, out_root,
                    stereo=args.stereo
                )
            )
# ────────────────────────────────────────────────────────────

    # --------------------------------------------------------
    # gather
    results = [f.result() for f in fut.as_completed(futures)]
    results = [r for r in results if r.get("ok")]

    # manifest へ書き込み
    man_file = Path(f"manifest_{args.split}.csv")
    header   = ["id", "pair_id", "pair_type",
                "caption", "mic4", "foa","stereo","meta"]
    write_header = not man_file.exists()

    with man_file.open("a", newline="", encoding="utf8") as fp:
        w = csv.DictWriter(fp, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(results)

    print(f"✅ {args.split}: {len(results)} samples written → {man_file}")


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
