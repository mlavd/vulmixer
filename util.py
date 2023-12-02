"""FIRE Utils Module.

This module provides utilities. Most of them are for loading models which are
not directly provided with this repository. Those functions have not been tested
thoroughly.
"""
from easydict import EasyDict as edict
import numpy as np
import yaml

def load_config(path):
    with open(path) as f:
        return edict(yaml.safe_load(f.read()))
    
def get_linevul(config):
    from models.linevul.linevul_model import Model as LineVul
    from transformers import RobertaConfig, RobertaForSequenceClassification

    model_config = RobertaConfig.from_pretrained(
        config.config,
        cache_dir=config.cache_dir,
    )
    model_config.max_position_embeddings = config.max_seq_length + 2

    model = RobertaForSequenceClassification.from_pretrained(
        config.model,
        config=model_config,
        cache_dir=config.cache_dir,
        ignore_mismatched_sizes=True,
    )

    for p in model.roberta.embeddings.parameters(): p.requires_grad = False
    for p in model.roberta.encoder.parameters(): p.requires_grad = False
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'trainable: {params:,d}')

    return LineVul(
        encoder=model,
        config=model_config,
        tokenizer=None,#tokenizer,
        args=None, # This argument is not used
        class_weights=None,
    )


def get_codebert(config):
    from models.codebert.model import Model as CodeBERT
    from transformers import RobertaConfig, RobertaForSequenceClassification

    model_config = RobertaConfig.from_pretrained(
        config.config,
        cache_dir=config.cache_dir,
    )
    model_config.max_position_embeddings = config.max_seq_length + 2

    model = RobertaForSequenceClassification.from_pretrained(
        config.model,
        config=model_config,
        cache_dir=config.cache_dir,
        ignore_mismatched_sizes=True,
    )

    for p in model.roberta.embeddings.parameters(): p.requires_grad = False
    for p in model.roberta.encoder.parameters(): p.requires_grad = False
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'trainable: {params:,d}')

    return CodeBERT(
        encoder=model,
        config=model_config,
        tokenizer=None,#tokenizer,
        args=None, # This argument is not used
        class_weights=None,
    )

def get_cotext(config):
    from transformers import AutoModelForSeq2SeqLM
    return AutoModelForSeq2SeqLM.from_pretrained(config.config)

def get_regvd(config):
    from models.regvd.model import GNNReGVD
    from transformers import RobertaConfig, RobertaForSequenceClassification

    model_config = RobertaConfig.from_pretrained(
        config.config,
        cache_dir=config.cache_dir,
    )
    model_config.max_position_embeddings = config.max_seq_length + 2

    model = RobertaForSequenceClassification.from_pretrained(
        config.model,
        config=model_config,
        cache_dir=config.cache_dir,
        ignore_mismatched_sizes=True,
    )

    from easydict import EasyDict as edict
    return GNNReGVD(
        encoder=model,
        config=model_config,
        tokenizer=None,#tokenizer,
        args=edict(
            gnn='ReGCN',
            feature_dim_size=768,
            hidden_size=128,
            num_GNN_layers=2,
            remove_residual=False,
            att_op='mul',
            num_classes=2,
            format='uni',
            window_size=5,
        ), # This argument is not used
        class_weights=None,
    )
