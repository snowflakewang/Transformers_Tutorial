import transformers
import random
import torch
import numpy as np
import os
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate
import pdb

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

class AFQMC(torch.utils.data.Dataset):
    def __init__(self, options, mode):
        if mode == 'Train':
            self.data = self.load_data('%s/train.json'%options.data_path)
        elif mode == 'Valid':
            self.data = self.load_data('%s/dev.json'%options.data_path)
        elif mode == 'Test':
            self.data = self.load_data('%s/dev.json'%options.data_path) # use dev.json temporarily
        else:
            raise Exception('[INFO] Invalid dataset type')
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collote_fn(batch_samples, tokenizer):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, # if only encode batch_sentence_1, then X['input_ids'].shape=[b, 54]
        batch_sentence_2, # if encode both, then X['input_ids'].shape=[b, 117]
        padding=True, # 补全
        truncation=True, # 裁剪
        return_tensors="pt" # 意为返回pytorch tensor
    )
    y = torch.tensor(batch_label)
    return X, y

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pairwise Classification')
    parser.add_argument('--config', type=str, default=None, help='config file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--base_model_ckpt', type=str, default='bert-base-chinese', help='ckpt for base model')
    parser.add_argument('--dataset', type=str, default='mrpc', help='dataset path')
    parser.add_argument('--save_path', type=str, default='model_pairwise_cls_ckpts', help='path to save new ckpts')
    options = parser.parse_args()

    seed_everything(options.seed)

    if options.config is not None:
        data = json.load(open(options.config, 'r'))
        for key in data:
            options.__dict__[key] = data[key]

    if options.dataset == 'mrpc':
        raw_dataset = load_dataset('glue', 'mrpc')
    elif options.dataset == 'afqmc_public':
        data_files = {"train": "data/afqmc_public/train.json", 
            "validation": "data/afqmc_public/dev.json",
             "test": "data/afqmc_public/dev.json"}
        raw_dataset = load_dataset('json', data_files=data_files)
    else:
        raise Exception('[INFO] Invalid dataset name')
    print('[INFO] Load %s dataset'%options.dataset)

    if options.base_model_ckpt == 'bert-base-chinese':
        checkpoints = '/cpfs01/user/wangyitong/.cache/huggingface/hub/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55'
    else:
        raise Exception('[INFO] Invalid base model checkpoints')
    tokenizer = AutoTokenizer.from_pretrained(checkpoints, model_max_length=512)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], padding=True, truncation=True)
    
    tokenized_raw = raw_dataset.map(tokenize_function, batched=True)
    # load_dataset() reads the labels of user-diy dataset as strings, which is not compatible
    # so we use list(map(int, examples['label']) to convert string to int
    tokenized_raw = tokenized_raw.map(lambda examples: {"label": list(map(int, examples['label']))}, batched=True)
    tokenized_raw = tokenized_raw.remove_columns(["sentence1", "sentence2"])
    tokenized_raw = tokenized_raw.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # pdb.set_trace()
    training_args = transformers.TrainingArguments(
        "train_model_ckpts/huggingface_trainer/%s"%options.dataset, 
        evaluation_strategy='epoch', 
        num_train_epochs=options.max_epoch,
        learning_rate=options.lr,
        optim='adamw_torch',
        # ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad']
        per_device_train_batch_size=options.batch_size,
        per_device_eval_batch_size=options.batch_size)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoints, num_labels=2)
    '''
    train_data = AFQMC(options, mode='Train')
    valid_data = AFQMC(options, mode='Valid')
    test_data = AFQMC(options, mode='Test')

    # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collote_fn(x, tokenizer))
    # valid_dataloader= DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collote_fn(x, tokenizer))
    # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collote_fn(x, tokenizer))
    # pdb.set_trace()
    '''
    trainer = transformers.Trainer(
    model,
    training_args,
    train_dataset=tokenized_raw['train'],
    eval_dataset=tokenized_raw['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )
    '''
    trainer = transformers.Trainer(
    model,
    training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=lambda x: collote_fn(x, tokenizer),
    tokenizer=tokenizer
    )
    '''

    trainer.train()