from pathlib import Path
from rich.progress import track
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Extract bigvul')
    parser.add_argument('--path', type=Path, default='./data/inputs/MSR_data_cleaned.csv', help='Path to Bigvul')
    parser.add_argument('--output', type=Path, default='./data/jsonl/bigvul', help='Path to output.')
    parser.add_argument('--vul', type=int, default=1, help='vul=1, safe=0')
    args = parser.parse_args()

    args.output = args.output.joinpath('vuln' if args.vul else 'safe')

    # Load CSV
    df = pd.read_csv(args.path)#, nrows=1)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Filter to vulnerabilities
    df = df[df.vul == args.vul]
    df = df[df.cve_id.notna()]
    print(f'# of Samples = {df.shape[0]:,d}')

    # Drop unnecessary columns
    df = df.drop(columns=[
        'unnamed:_0', 'access_gained', 'cve_page', 'publish_date', 'summary', 'update_date',
        'add_lines', 'codelink', 'commit_id', 'commit_message', 'del_lines', 'file_name',
        'files_changed',  'vul', 'vul_func_with_fix', 'project_before', 'project_after'
    ])

    # Add "idx"
    df['idx'] = range(df.shape[0])

    # # Save to JSON
    # df.to_json(str(args.output) + '.jsonl', orient='records', lines=True)

    # Save before only
    before = df.copy()
    before['func'] = before.func_before
    before['target'] = args.vul
    before = before.drop(columns=['func_before', 'func_after'])
    before.to_json(str(args.output) + '_before.jsonl', orient='records', lines=True)

    # Save after only
    after = df.copy()
    after['func'] = after.func_after
    after['target'] = 0
    after = after.drop(columns=['func_before', 'func_after'])
    after.to_json(str(args.output) + '_after.jsonl', orient='records', lines=True)

    print('Done')
