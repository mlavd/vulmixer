"""FIRE Transforms Module.

This module provides different dataset transformations for various models.
"""
import torch
import torch.nn.functional as F

class Transform:
    def transform(self):
        raise NotImplementedError()

    def __call__(self, x):
        return [ self.transform(xi) for xi in x ]

class FireTransform(Transform):
    # FIXME Handle projections and document
    def __init__(self, checkpoint, max_seq_length, num_features, truncate, normalize):
        from transformers import AutoTokenizer

        self.max_seq_length = max_seq_length
        self.num_features = num_features
        self.truncate = truncate
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, use_auth_code=True)
        
        # FIXME Make this configurable
        name = checkpoint.split('/')[-1] + '.embeddings'
        self.embedding = torch.load(f'./data/cache/{name}').to('cuda')        

    def transform(self, x):
        if self.normalize:
            raise Exception('Normalizing is not supported yet.')
            # x = super_norm(x)

        tokens = self.tokenizer.encode(x)[:self.max_seq_length]
        embedding = self.embedding(torch.tensor(tokens, device='cuda')).detach()

        if self.truncate:
            embedding = embedding[:, :self.num_features]

        return F.pad(embedding, (0, 0, 0, self.max_seq_length - len(tokens)), 'constant', 0)
    
    def __call__(self, x):
        return torch.stack([ self.transform(xi) for xi in x ])


class TextCNNTransform(Transform):
    def __init__(self, checkpoint, max_seq_length):
        from transformers import AutoTokenizer
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, use_auth_code=True)       

    def transform(self, x):
        tokens = self.tokenizer.encode(x,
            padding='max_length', max_length=self.max_seq_length, truncation=True)
        return tokens

class CodeBERTTransform(Transform):
    def __init__(self, tokenizer, cache_dir, max_seq_length):
        from transformers import RobertaTokenizer

        self.max_seq_length = max_seq_length
        self.tokenizer = RobertaTokenizer.from_pretrained(
            tokenizer,
            cache_dir=cache_dir
        )
    
    def transform(self, x):
        x = super_norm(x)

        return self.tokenizer.encode(
            x,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
        )

class CoTexTTransform(Transform):
    def __init__(self, tokenizer, cache_dir, max_seq_length):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir=cache_dir)
        self.max_seq_length = max_seq_length

    def transform(self, x):
        inputs = 'defect_detection: ' + x
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
        )

        # # Setup the tokenizer for targets
        # with self.tokenizer.as_target_tokenizer():
        #     # 2 tokens is enough for the 'true' or 'false'
        #     text_labels = [ 'true' if e == 1 else 'false' for e in x['target'] ]
        #     labels = self.tokenizer(text_labels, max_length=2, truncation=True)

        # model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def get_transforms(config):
    # FIXME Move to lib
    if config.type in [ 'pnlp', 'tcam', 'fnet', 'fire' ]:
        return FireTransform(
            checkpoint=config.input.embeddings,
            max_seq_length=config.max_seq_length,
            num_features=config.input.embedding_size,
            truncate=config.input.truncate,
            normalize=config.input.normalize,
        )

    elif config.type == 'textcnn':
        return TextCNNTransform(
            checkpoint=config.embeddings,
            max_seq_length=config.max_seq_length,
        )

    elif config.type in [ 'codebert', 'regvd', 'linevul' ]:
        return CodeBERTTransform(
            cache_dir=config.cache_dir,
            max_seq_length=config.max_seq_length,
            tokenizer=config.tokenizer,
        )

    elif config.type == 'cotext':
        return CoTexTTransform(
            cache_dir=config.cache_dir,
            max_seq_length=config.max_seq_length,
            tokenizer=config.tokenizer,
        )

    else:
        raise NotImplementedError(f'Invalid type: {config.type}')