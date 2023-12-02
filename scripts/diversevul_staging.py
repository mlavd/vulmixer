#!/usr/bin/env python
"""Format JSONL datasets with clang-format

This requires clang-format >= 15 because it uses the 
"""
from pathlib import Path
import argparse
import pandas as pd
import json
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path,
                        default='./data/inputs/diversevul_20230702.json',
                        help='JSONL input')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=int, default=0.1)
    args = parser.parse_args()
    print(args)

    print('Loading DiverseVul')
    with open(args.input, 'r') as f:
        lines = [ json.loads(l) for l in f ]
        df = pd.DataFrame(lines)

    
    print('Splitting')
    df_train, df_test = train_test_split(
        df[['func', 'target']],
        stratify=df.target,
        test_size=0.2,
        random_state=args.seed,
    )

    df_test, df_valid = train_test_split(
        df_test,
        stratify=df_test.target,
        test_size=0.5,
        random_state=args.seed,
    )

    df_train['idx'] = range(df_train.shape[0])
    df_test['idx'] = range(df_test.shape[0])
    df_valid['idx'] = range(df_valid.shape[0])

    print(df_train.target.value_counts())
    print(df_valid.target.value_counts())
    print(df_test.target.value_counts())


    print('Saving')
    df.to_json('./data/jsonl/diversevul/all.jsonl', orient='records', lines=True)
    df_train.to_json('./data/jsonl/diversevul/train.jsonl', orient='records', lines=True)
    df_valid.to_json('./data/jsonl/diversevul/valid.jsonl', orient='records', lines=True)
    df_test.to_json('./data/jsonl/diversevul/test.jsonl', orient='records', lines=True)

    print('Done')
