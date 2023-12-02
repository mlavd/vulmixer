"""DiverseVul Testing Module.

This module runs a model against the DiverseVul dataset. See arguments for
configuration parameters.
"""
from cl import CurriculumLearning
from rich.console import Console
from run import get_transforms, get_model
from transforms import *
from util import load_config
import argparse
import lightning.pytorch as pl
import pandas as pd
import torch
from sklearn import metrics

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


    # Get the data module
    # from dataset import JSONLDataModule
    # data_module = CurriculumDataModule(
    #     cl_config=cl_config,
    #     # data_config=data_config,
    #     batch_size=model_config.batch_size,
    #     num_workers=args.num_workers,
    # )
    # console.log('class_weights=', data_module.get_class_weights())


    # FIXME Handle checkpoints


    # Get the models
    model = get_model(model_config)
    console.log('[bold green]Loaded model')
    console.log(f'model=[purple]{model._get_name()}')

    if args.checkpoint:
        # FIXME make this more pytorch lightning-ish
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
        calculate_metrics=True,
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
                'target': row.target,
                'weight': torch.Tensor(1),
                'input': row.func
            } for i, row in df.iterrows()
        ])),
        batch_size=32,
        shuffle=False,
        num_workers=1,
    )

    trainer.test(train_module, befores)
    df['y_pred'] = train_module.test_preds.copy()
    df[['idx', 'y_pred']].to_csv('diversevul.csv', index=False)

    THRESH = 0.5
    print(f'AP       : {metrics.average_precision_score(df.target, df.y_pred):.4f}')
    print(f'Accuracy : {metrics.accuracy_score(df.target, df.y_pred > THRESH):.4f}')
    print(f'F1       : {metrics.f1_score(df.target, df.y_pred > THRESH):.4f}')
    print(f'Precision: {metrics.precision_score(df.target, df.y_pred > THRESH):.4f}')
    print(f'Recall   : {metrics.recall_score(df.target, df.y_pred > THRESH):.4f}')
    print('-----')

    tn, fp, fn, tp = metrics.confusion_matrix(df.target, df.y_pred > THRESH).ravel()
    print(f'TPs: {tp}')
    print(f'FPs: {fp}')
    print(f'TNs: {tn}')
    print(f'FNs: {fn}')


if __name__ == '__main__': 
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, required=True)
    args.add_argument('--cl', type=str, default='none')
    args.add_argument('--checkpoint', type=str, default=None)
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--dataset', type=str, default='./data/jsonl/diversevul/all.jsonl')
    args = args.parse_args()
    main(args)    
