#!/usr/bin/env python3
# gen_room_pool.py  –  train/val 用 8952 部屋を生成し、
#                      test 範囲を満たす 4970 部屋サブセットを抜き出し、
#                      最終的に trainval を 8952 件に切り詰め
#
#   出力:
#     room_pool_trainval.json   (8952 部屋)
#     room_pool_test.json       (4970 部屋; trainval の subset)
#
#   使い方:
#     python scripts/gen_room_pool.py

import random, json, yaml, math
from pathlib import Path

# 設定読み込み
cfg = yaml.safe_load(Path("spatial_ranges.yml").read_text())
TV_N   = cfg["TRAINVAL_ROOMS"]  # 8952
TEST_N = cfg["TEST_ROOMS"]      # 4970

# train/val 範囲
rng_tv = cfg["TRAINVAL"]
# test 範囲
rng_t  = cfg["TEST"]

def gen_one_room():
    """ランダムに 1 部屋を生成（面積→寸法→体積→表面積→T30→吸音率 alpha）"""
    area = random.uniform(rng_tv["AREA_MIN"], rng_tv["AREA_MAX"])
    aspect = random.uniform(1.0,2.5)
    w = h = math.sqrt(area*aspect)
    h = area/w
    H = 3.0
    V = w * h * H
    S = 2*(w*h + w*H + h*H)

    # T30 はミリ秒でランダム抽出
    T30_ms = random.uniform(rng_tv["T30_MIN"], rng_tv["T30_MAX"])
    T30 = T30_ms / 1000.0
    # Sabineの式から alpha を逆算, クリップ
    alpha = 0.161 * V / (S * T30)
    alpha = max(cfg["ABS_MIN"], min(cfg["ABS_MAX"], alpha))

    return dict(
        dims=[round(w,3), round(h,3), round(H,3)],
        area_m2=round(area,2),
        alpha=round(alpha,4),
        T30_ms=round(T30_ms,1)
    )

def in_test_range(r):
    """部屋 r の area_m2 と T30_ms が test 範囲内かを判定"""
    return (
        rng_t["AREA_MIN"] <= r["area_m2"] <= rng_t["AREA_MAX"]
        and rng_t["T30_MIN"] <= r["T30_ms"] <= rng_t["T30_MAX"]
    )

# メイン生成ループ
trainval = []
test_candidates = []

# trainval が TV_N 件＆ test_candidates が TEST_N 件揃うまで回す
while len(trainval) < TV_N or len(test_candidates) < TEST_N:
    room = gen_one_room()
    trainval.append(room)
    if in_test_range(room):
        test_candidates.append(room)

# test_pool を抜き出し（trainval のサブセット）
test_pool = random.sample(test_candidates, TEST_N)

# trainval プールをピッタリ TV_N 件に切り詰める
# └ test_pool は必ず含め、残りをランダム抽出
others = [r for r in trainval if r not in test_pool]
needed = TV_N - TEST_N
assert needed >= 0, "TEST_ROOMS が TRAINVAL_ROOMS を超えています"
chosen = random.sample(others, needed)  # 残り部屋を抽出
final_trainval = test_pool + chosen
random.shuffle(final_trainval)

# JSON 出力
Path("room_pool_trainval.json").write_text(json.dumps(final_trainval, indent=2))
Path("room_pool_test.json").write_text(json.dumps(test_pool, indent=2))

print(f"✅ room_pool_trainval.json : {len(final_trainval)} rooms")
print(f"✅ room_pool_test.json     : {len(test_pool)} rooms  (subset)")
