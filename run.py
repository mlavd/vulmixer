"""FIRE training module.

The Framework for Investigating Resource-Efficent (FIRE) MLAVD makes training
and testing MLAVD models easy. It is customized through configuration files
for each model and dataset.

This is the primary training script. See README.md for examples of usage.

As you read the file, you may notice some code which is not used in the paper.
This code includes options for curriculm learning, code normalization, and
expanding to additional models. We have opted to leave this code in place to
assist with future development.
"""
from cl import CurriculumLearning
from cl_dataset import CurriculumDataModule
from pathlib import Path
from rich.console import Console
from transforms import *
from util import get_codebert, get_linevul, get_regvd, load_config, get_cotext
import argparse
import lightning.pytorch as pl
import torch

def get_model(config):
    """Gets the appropriate model from the configuration file."""
    if config.type == 'pnlp':
        from models.pnlp.model import PnlpMixerSequenceClassifier
        return PnlpMixerSequenceClassifier(config)

    elif config.type == 'tcam':
        from models.tcam.model import TCAMixerSequenceClassifier
        return TCAMixerSequenceClassifier(config)

    elif config.type == 'fire':
        from models.fire.model import FireSVD
        return FireSVD(config)
    
    elif config.type == 'fnet':
        from models.fnet.model import FNetSequenceClassifier
        return FNetSequenceClassifier(config)
    
    elif config.type == 'textcnn':
        from models.textcnn.model import TextCNN
        return TextCNN(
            checkpoint=config.embeddings,
            kernel_sizes=config.kernel_sizes,
            num_filters=config.num_filters,
            dropout=config.dropout,
            mode=config.mode,
        )

    elif config.type == 'codebert':
        return get_codebert(config)

    elif config.type == 'regvd':
        return get_regvd(config)
    
    elif config.type == 'linevul':
        return get_linevul(config)

    elif config.type == 'cotext':
        return get_cotext(config)

    else:
        raise NotImplementedError(f'Invalid type: {config.type}')


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

    # Get the models
    model = get_model(model_config)
    console.log('[bold green]Loaded model')
    console.log(f'model=[purple]{model._get_name()}')

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        state_dict = {
            (
                k.replace('local_information', 'statement_info')
                 .replace('global_information', 'function_info')
            ): v
            for k, v in checkpoint['state_dict'].items()
        }
        model.load_state_dict({ k[len('model.'):]: v for k, v in state_dict.items() })
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
    )

    # Setup MLFlow logger
    name = f'm={args.model}'#, cl={args.cl}, s={cl_config.schedule}'
    if args.suffix: name += ', ' + args.suffix
    logger = None

    if args.log.stem != 'None' and args.do_train:
        logger = [
            pl.loggers.MLFlowLogger(
                experiment_name = args.data,
                run_name = name,
                save_dir = args.log.absolute(),
#                log_model=False,
            )
        ]
    
        for l in logger:
            l.log_hyperparams(params=dict(
                batch_size = model_config.batch_size,
                model  = args.model,
                learning_rate = model_config.optimizer.lr,
                all_params = sum(p.numel() for p in model.parameters()),
                train_params = sum(p.numel() for p in model.parameters() if p.requires_grad),
            ))


    # Set precision
    torch.set_float32_matmul_precision('medium')

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_ap',
        save_last=True,
        save_top_k=2,
        mode='max',
        save_weights_only=True,
        auto_insert_metric_name=True,
    )

    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=args.max_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            checkpoint_cb,
            pl.callbacks.RichProgressBar(leave=True),
        ],
        logger=logger,
    )


    if args.do_train:
        trainer.fit(train_module, data_module)

        print(checkpoint_cb.best_model_path)  # prints path to the best model's checkpoint
        print(checkpoint_cb.best_model_score) # and prints it score

        checkpoint = torch.load(checkpoint_cb.best_model_path)
        model.load_state_dict({ k[len('model.'):]: v for k, v in checkpoint['state_dict'].items() })
        console.log('Loaded checkpoint.')
        console.log(f'checkpoint=[purple]{args.checkpoint}')


    if args.do_test:
        trainer.test(train_module, data_module)

        with open('predictions.txt', 'w') as f:
            for index, pred in train_module.test_preds.items():
                f.write(f'{index}\t{pred:.4f}\n')



if __name__ == '__main__': 
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, required=True)
    args.add_argument('--model', type=str, required=True)
    args.add_argument('--cl', type=str, default='none')
    args.add_argument('--suffix', type=str, default='')
    args.add_argument('--checkpoint', type=str, default=None)
    args.add_argument('--do_train', action='store_true')
    args.add_argument('--do_test', action='store_true')
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--max_epochs', default=20, type=int)
    args.add_argument('--num_workers', default=8, type=int)
    args.add_argument('--log', default='./logs/mlflow', type=Path)
    args = args.parse_args()
    main(args)    
