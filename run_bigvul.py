"""Big-Vul Testing Module.

This module runs a model against the Big-Vul dataset. See arguments for
configuration parameters.
"""
from cl import CurriculumLearning
from datetime import timedelta
from rich.console import Console
from run import get_transforms, get_model
from transforms import *
from util import load_config
import argparse
import lightning.pytorch as pl
import pandas as pd
import torch

def main(args):
    # Setup a ring console and print the args.
    console = Console()
    console.log('[bold green]Start run')
    console.log('args=', args)


    # Load configurations
    model_config = load_config(f'config/model/{args.model}.yml')
    cl_config = load_config(f'config/cl/{args.cl}.yml')
    console.log('[bold green]Loaded Configs')
    console.log('model_config=', model_config)
    console.log('cl_config=', cl_config)

    # Seed everything
    pl.seed_everything(42)



    # Get the models
    model = get_model(model_config)
    console.log('[bold green]Loaded model')
    console.log(f'model=[purple]{model._get_name()}')

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)

        if 'state_dict' in checkpoint:
            weights = {
                k[len('model.'):].replace('local_information', 'statement_info')
                 .replace('global_information', 'function_info'): v
                for k, v in checkpoint['state_dict'].items()
            }
            
        else:
            # Older models don't need to remove the "model."
            weights = checkpoint

        model.load_state_dict(weights)
        console.log('Loaded checkpoint.')
        console.log(f'checkpoint=[purple]{args.checkpoint}')


    # Get transforms
    transforms = get_transforms(model_config)

    # Build the LFF Module
    train_module = CurriculumLearning(
        model=model,
        transform=transforms,
        config=cl_config,
        weights=torch.Tensor([1, 1]),
        device=torch.device(args.device),
        model_config=model_config,
        calculate_metrics=False,
    )

    # Set precision
    torch.set_float32_matmul_precision('medium')


    timer = pl.callbacks.Timer()

    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=1,
        reload_dataloaders_every_n_epochs=1,
        callbacks = [ timer ],
        check_val_every_n_epoch = 9999, # Never check it
        # profiler='pytorch',
    )

    df = pd.read_json(args.dataset, lines=True)
    print(df.shape)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]


    befores = torch.utils.data.DataLoader(
        dataset=Dataset(list([
            {
                'index': i,
                'target':torch.Tensor([1, 0]),
                'weight': torch.Tensor(1),
                'input': row.func_before
            } for i, row in df.iterrows()
        ])),
        batch_size=32,
        shuffle=False,
        num_workers=1,
    )

    trainer.test(train_module, befores)
    before_preds = train_module.test_preds.copy()

    afters = torch.utils.data.DataLoader(
        dataset=Dataset(list([
            {
                'index': i,
                'target':torch.Tensor([0, 1]),
                'weight': torch.Tensor(1),
                'input': row.func_after,
            } for i, row in df.iterrows()
        ])),
        batch_size=32,
        shuffle=False,
        num_workers=1,
    )
    trainer.test(train_module, afters)
    after_preds = train_module.test_preds.copy()


    elapsed = timer.time_elapsed('test')
    delta = timedelta(seconds=elapsed)
    print('Elapsed Time:', delta)
    print('Raw Seconds:', elapsed)
    console.log('Done')

    preds = pd.DataFrame({'before': before_preds, 'after': after_preds})
    preds.to_csv('bigvul.csv', index=False)

    preds['diff'] = preds.before - preds.after
    print('---------- Mean ----------')
    print(preds.mean())

    print('---------- Median ----------')
    print(preds.median())

    print('---------- STD ----------')
    print(preds.std())

    # if args.do_test:
    #     trainer.test(train_module, data_module)

    #     with open('predictions.txt', 'w') as f:
    #         for index, pred in train_module.test_preds.items():
    #             f.write(f'{index}\t{pred:.4f}\n')



if __name__ == '__main__': 
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, required=True)
    args.add_argument('--cl', type=str, default='none')
    args.add_argument('--checkpoint', type=str, default=None)
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--dataset', type=str, default='./data/jsonl/bigvul/vuln.jsonl')
    args = args.parse_args()
    main(args)    
