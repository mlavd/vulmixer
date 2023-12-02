from pathlib import Path
from rich.progress import track
import argparse
import h5py
import pandas as pd

def read_hdf5(file):
    dictionary = {}
    with h5py.File(file, "r") as f:
        for key in f.keys():
            dictionary[key] = f[key][()]

    return pd.DataFrame.from_dict(dictionary)

def run_task(args):
    # Setup base paths
    base_out = Path(f'./data/jsonl/draper/')
    base_out.mkdir(exist_ok=True, parents=True)

    base_in = Path('./data/inputs/draper')
    hdf5_files = list(base_in.glob('*.hdf5'))

    # Progress bar
    for hdf5_file in track(hdf5_files):
        df = read_hdf5(hdf5_file)
        cwes = [ c for c in df.columns if c.startswith('CWE') ]
        df['target'] = df[cwes].any(axis=1).astype(int)
        df['func'] = df['functionSource'].str.decode('utf-8')
        df = df[['target', 'func']].reset_index().rename(columns={'index': 'idx'})

        split_name = hdf5_file.stem.rpartition('_')[2]
        split_path = base_out.joinpath(f'{split_name}.jsonl')

        df.to_json(split_path, orient='records', lines=True)
        print(f'{split_name}: {df["target"].sum()}')

    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='JSONL Draper Exporter')
    args = parser.parse_args()
    run_task(args)