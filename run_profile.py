"""Profiling Module.

This module runs profiles a model. See arguments for configuration parameters.
"""
from cl import CurriculumLearning
from cl_dataset import CurriculumDataModule
from datetime import timedelta
from rich.console import Console
from transforms import *
from util import load_config
import argparse
import lightning.pytorch as pl
import torch

from run import get_transforms, get_model


def main(args):
    # Setup a ring console and print the args.
    console = Console()
    console.log('[bold green]Start run')
    console.log('args=', args)


    # Load configurations
    data_config = load_config(f'config/data/{args.data}.yml')
    model_config = load_config(f'config/model/{args.model}.yml')
    cl_config = load_config(f'config/cl/{args.cl}.yml')
    console.log('[bold green]Loaded Configs')
    console.log('data_config=', data_config)
    console.log('model_config=', model_config)
    console.log('cl_config=', cl_config)

    # Seed everything
    pl.seed_everything(42)


    # Get the data module
    # from dataset import JSONLDataModule
    data_module = CurriculumDataModule(
        cl_config=cl_config,
        data_config=data_config,
        batch_size=model_config.batch_size,
        num_workers=args.num_workers,
    )
    console.log('class_weights=', data_module.get_class_weights())


    # FIXME Handle checkpoints


    # Get the models
    model = get_model(model_config)
    console.log('[bold green]Loaded model')
    console.log(f'model=[purple]{model._get_name()}')

    if args.checkpoint:
        # FIXME make this more pytorch lightning-ish
        checkpoint_cb = torch.load(args.checkpoint)
        model.load_state_dict({ k.lstrip('model.'): v for k, v in checkpoint_cb['state_dict'].items() })
        console.log('Loaded checkpoint.')
        console.log(f'checkpoint=[purple]{args.checkpoint}')


    # Get transforms
    transforms = get_transforms(model_config)

    # Build the LFF Module
    train_module = CurriculumLearning(
        model=model,
        transform=transforms,
        config=cl_config,
        weights=data_module.get_class_weights(),
        device=torch.device(args.device),
        model_config=model_config,
        calculate_metrics=False,
    )

    # Set precision
    torch.set_float32_matmul_precision('medium')


    timer = pl.callbacks.Timer()

    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks = [ timer ],
        check_val_every_n_epoch = 9999, # Never check it
        # gradient_clip_val=0.5, # NOTE If you have issues with FNet nan-loss, enable clipping
        # profiler='pytorch',
    )


    if args.profile == 'train':
        trainer.fit(train_module, data_module)

    elif args.profile == 'test':
        trainer.test(train_module, data_module)

    elapsed = timer.time_elapsed(args.profile)
    delta = timedelta(seconds=elapsed)
    print('Elapsed Time:', delta)
    print('Raw Seconds:', elapsed)
    console.log('Done')



if __name__ == '__main__': 
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, required=True)
    args.add_argument('--cl', type=str, default='none')
    args.add_argument('--data', default='profile', type=str)
    args.add_argument('--max_epochs', default=1, type=int)
    args.add_argument('--num_workers', default=8, type=int)
    # args.add_argument('--log', default='./logs/mlflow', type=Path)
    args.add_argument('--checkpoint', type=str, default=None)
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--profile', default='train', type=str)
    args = args.parse_args()
    main(args)    
