import random
import os
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm

import argparse
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

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
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

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, # 补全
        truncation=True, # 裁剪
        return_tensors="pt" # 意为返回pytorch tensor
    )
    y = torch.tensor(batch_label)
    return X, y

class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()
    
    def forward(self, x):
        outputs = self.bert(**x) # x['input_ids'].shape=[4, 39]
        # using last_hidden_state means we do not use the original classifier and initialize a new one
        cls_vectors = outputs.last_hidden_state[:, 0, :] # outputs.last_hidden_state.shape=[4, 39, 768]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)
    
    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(options, dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    ckpts = sorted(os.listdir(options.save_path))
    model.load_state_dict(torch.load('%s/%s'%(options.save_path, ckpts[-1])))
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pairwise_classification')
    parser.add_argument('--config', type=str, default=None, help='config file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--base_model_ckpt', type=str, default='bert-base-chinese', help='ckpt for base model')
    parser.add_argument('--data_path', type=str, default='data/afqmc_public', help='dataset path')
    parser.add_argument('--save_path', type=str, default='model_pairwise_cls_ckpts', help='path to save new ckpts')
    options = parser.parse_args()

    if options.config is not None:
        data = json.load(open(options.config, 'r'))
        for key in data:
            options.__dict__[key] = data[key]

    os.makedirs('train_model_ckpts', exist_ok=True)
    options.save_path = 'train_model_ckpts/' + options.save_path
    os.makedirs(options.save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    seed_everything(options.seed)

    learning_rate = options.lr
    batch_size = options.batch_size
    epoch_num = options.max_epoch

    checkpoint = options.base_model_ckpt
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    train_data = AFQMC('%s/train.json'%options.data_path)
    valid_data = AFQMC('%s/dev.json'%options.data_path)
    test_data = AFQMC('%s/test.json'%options.data_path)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
    valid_dataloader= DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

    config = AutoConfig.from_pretrained(checkpoint)
    model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num*len(train_dataloader),
    )

    total_loss = 0.
    best_acc = 0.
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
        valid_acc = test_loop(valid_dataloader, model, mode='Valid')
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'{options.save_path}/epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.pt')
    
    test_loop(options, test_dataloader, model, mode='Test')