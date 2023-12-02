#!/usr/bin/env python
"""Format JSONL datasets with clang-format

This requires clang-format >= 15 because it uses the 
"""
from pathlib import Path
from rich.progress import track
import argparse
import pandas as pd
import subprocess

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, help='JSONL input')
    parser.add_argument('--col', type=str, default='func', help='Column to format.')
    args = parser.parse_args()
    print(args)

    print('Loading JSONL')
    df = pd.read_json(args.input, lines=True)
    print(df.columns)

    for i in track(range(df.shape[0])):
        try:
            output = subprocess.check_output(
                ['clang-format', '-style=file:config/clang-format.yml'],
                input=df.loc[i, args.col],
                text=True,
                timeout=15
            )
            df.loc[i, args.col] = output
        except subprocess.CalledProcessError as e:
            print('CalledProcessError')
            print(e)
            exit(-1)
        except subprocess.TimeoutExpired as e:
            print('TimeoutExpired')
            print(i)
            exit(-1)

    print('Saving')
    print(df.shape)
    df.to_json(args.input, orient='records', lines=True)

    print('Done')
