from pathlib import Path
from rich.progress import track
import argparse
import pandas as pd

def run_task(args):
    base_out = Path(f'./data/jsonl/d2a/')
    base_out.mkdir(exist_ok=True, parents=True)

    base_in = Path('./data/inputs/d2a')
    print(list(base_in.glob('*.csv')))

    for split_file in track(base_in.glob('*.csv')):
        print(split_file)
        _, _, split_name = split_file.stem.rpartition('_')
        df = pd.read_csv(split_file)
        df['label'] = df['label'] if 'label' in df.columns else 0
        df = df[['label', 'code']]
        df.columns = ['target', 'func']
        df = df.reset_index().rename(columns={'index': 'idx'})
        df.to_json(base_out.joinpath(f'{split_name}.jsonl'), orient='records', lines=True)

    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='JSONL D2A Staging')
    args = parser.parse_args()
    run_task(args)