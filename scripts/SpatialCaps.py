#!/usr/bin/env python3
"""
SpatialCaps.py: Rewrite a spatial audio caption using OpenAI's Chat API.

Usage (Windows PowerShell):
  # Set your API key in PowerShell
  $Env:OPENAI_API_KEY = "your_api_key_here"
  python SpatialCaps.py --meta output/meta.yml --caption "Someone crumples paper"

Alternative: derive original caption from a CSV mapping of IDs (e.g., train.csv):
  python SpatialCaps.py --meta output/meta.yml --id 91139 --csv train.csv

Dependencies:
  pip install openai pyyaml pandas
"""

import openai
openai.api_key = "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"
import os
import argparse
import yaml
import pandas as pd
import openai
import sys
def load_meta(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def map_direction(az):
    # azimuth in degrees
    # check front, left, right, then back ([-145,+45])
    if -35.0 <= az <= 35.0:
        return 'front'
    elif 55.0 <= az <= 125.0:
        return 'right'
    elif -125.0 <= az <= -55.0:
        return 'left'
    elif -145.0 <= az <= 45.0:
        return 'back'
    else:
        return ''

def map_elevation(ele):
    # elevation in degrees
    if ele < -40.0:
        return 'down'
    elif ele > 40.0:
        return 'up'
    else:
        return ''
def map_distance(dist):
    # dist in meters
    if dist < 1.0:
        return 'near'
    elif dist > 2.0:
        return 'far'
    else:
        return ''


def map_size(area):
    # area in m2
    if area < 50.0:
        return 'small'
    elif area > 100.0:
        return 'large'
    else:
        return 'mid-sized'

def map_reverb(t30):
    # t30 in ms
    if t30 < 200.0:
        return 'acoustically dampened'
    elif t30 > 1000.0:
        return 'highly reverberant'
    else:
        return ''

def rewrite_spatial_caption(original, meta):
    dist = meta.get('source_distance_m')
    az_deg = meta.get('azimuth_deg')
    ele_deg = meta.get('elevation_deg')

    room_floor_m2 = meta.get('room_floor_m2') 
    t30 = meta.get('fullband_T30_ms')

    distance = map_distance(dist)
    direction = map_direction(az_deg)
    elevation = map_elevation(ele_deg)
    size = map_size(room_floor_m2)
    reverb = map_reverb(t30)

    # Build prompt
    # e.g.: The sound: "<orig>" is coming from the <dir> <dist> of a <size> room <reverb_desc>.
    desc_parts = [direction]
    if distance:
        desc_parts.append(distance)
    location_desc = ' '.join(desc_parts)

    room_desc = f'a {size} room'
    if reverb:
        room_desc += f' that is {reverb}'

    prompt = (
        f'The sound: "{original}" is coming from the {distance} {elevation} {direction} of a {size} {reverb} room. '
        'Rephrase as a short English sentence describing the sound and all details of its source.'
    )

    # Call OpenAI Chat API (v1.0+ interface) ([note.com](https://note.com/gorojy/n/n5e37bf9df9b0?utm_source=chatgpt.com))
    response = openai.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        temperature = 0.9,
        max_tokens = 1024
    )
    # access new content
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description='Rewrite spatial audio captions')
    parser.add_argument('--meta', required=True, help='Path to the meta YAML file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--caption', help='Original caption text')
    group.add_argument('--id', help='ID to lookup in CSV')
    parser.add_argument('--csv', help='Optional CSV file mapping ID to caption')
    args = parser.parse_args()

    meta = load_meta(args.meta)

    if args.caption:
        original = args.caption
    else:
        if not args.csv:
            print('Error: --csv must be provided when using --id', file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.csv, dtype={'audiocap_id': str})
        df.set_index('audiocap_id', inplace=True)
        try:
            original = df.loc[args.id]['caption']
        except KeyError:
            print(f'ID {args.id} not found in {args.csv}', file=sys.stderr)
            sys.exit(1)

    new_caption = rewrite_spatial_caption(original, meta)
    print(new_caption)

if __name__ == '__main__':
    main()
