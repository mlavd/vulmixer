from transformers import AutoModel
from pathlib import Path
import argparse
import torch

def main(args):
    model = AutoModel.from_pretrained(
        args.checkpoint)

    out = args.cache_dir.joinpath(args.checkpoint.split('/')[-1] + '.embeddings')
    
    if args.checkpoint in [
        'microsoft/codebert-base', 'microsoft/unixcoder-base']:
        embeddings = model.roberta.embeddings.word_embeddings
    
    elif args.checkpoint in [
        'microsoft/graphcodebert-base', 'microsoft/longcoder-base' ]:
        embeddings = model.embeddings.word_embeddings
    
    else:
        raise NotImplementedError('Checkpoint not supported')

    torch.save(embeddings, out)
    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='HF Checkpoint')
    parser.add_argument('--cache_dir', type=Path, default='./data/cache')
    args = parser.parse_args()
    main(args)

