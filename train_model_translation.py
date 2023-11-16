import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from sacrebleu.metrics import BLEU
from tqdm.auto import tqdm
import json

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class TRANS(Dataset):
    def __init__(self, options, mode):
        if mode == 'train':
            self.data = self.load_data('%s/translation2019zh_train.json'%options.data_path)
        elif mode == 'test':
            self.data = self.load_data('%s/translation2019zh_valid.json'%options.data_path)
        else:
            raise Exception('[INFO] Invalid dataset type')
        
        self.options = options
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= options.max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collote_fn(batch_samples, tokenizer):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=options.max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=options.max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

bleu = BLEU()

def test_loop(dataloader, model):
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
            ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f"BLEU: {bleu_score:>0.2f}\n")
    return bleu_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine Translation')
    parser.add_argument('--config', type=str, default=None, help='config file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--base_model_ckpt', type=str, default='Helsinki-NLP/opus-mt-zh-en', help='ckpt for base model')
    parser.add_argument('--data_path', type=str, default='data/translation2019zh', help='dataset path')
    parser.add_argument('--save_path', type=str, default='model_translation_ckpts', help='path to save new ckpts')
    options = parser.parse_args()

    if options.config is not None:
        data = json.load(open(options.config, 'r'))
        for key in data:
            options.__dict__[key] = data[key]

    os.makedirs('train_model_ckpts', exist_ok=True)
    options.save_path = 'train_model_ckpts/' + '%s_%s'%(options.save_path, options.base_model_ckpt.replace('/', '-'))
    os.makedirs(options.save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    seed_everything(options.seed)

    max_dataset_size = options.max_dataset_size

    max_input_length = options.max_input_length
    max_target_length = options.max_target_length

    batch_size = options.batch_size
    learning_rate = options.lr
    epoch_num = options.max_epoch

    data = TRANS(options, mode='train')

    if options.train_ratio > 0 and options.train_ratio < 1:
        train_set_size = int(options.train_ratio * data.__len__())
    else:
        train_set_size = int(0.9 * data.__len__())
    valid_set_size = data.__len__() - train_set_size

    train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
    test_data = TRANS(options, mode='test')

    model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model = model.to(device)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collote_fn(x, tokenizer))
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collote_fn(x, tokenizer))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collote_fn(x, tokenizer))

    bleu = BLEU()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num*len(train_dataloader),
    )

    total_loss = 0.
    best_bleu = 0.
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
        valid_bleu = test_loop(valid_dataloader, model)
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            print('saving new weights...\n')
            torch.save(model.state_dict(), '%s/%s_finetuned_weights.pt'%(options.save_path, options.base_model_ckpt))
            with open('%s/log.txt'%options.save_path, 'w') as w:
                w.write('best checkpoint epoch: %d\n'%(t + 1))
                w.write('best checkpoint acc: %.05f'%best_acc)

# import json

# model.load_state_dict(torch.load('epoch_1_valid_bleu_53.38_model_weights.bin'))

# model.eval()
# with torch.no_grad():
#     print('evaluating on test set...')
#     sources, preds, labels = [], [], []
#     for batch_data in tqdm(test_dataloader):
#         batch_data = batch_data.to(device)
#         generated_tokens = model.generate(
#             batch_data["input_ids"],
#             attention_mask=batch_data["attention_mask"],
#             max_length=max_target_length,
#         ).cpu().numpy()
#         label_tokens = batch_data["labels"].cpu().numpy()

#         decoded_sources = tokenizer.batch_decode(
#             batch_data["input_ids"].cpu().numpy(), 
#             skip_special_tokens=True, 
#             use_source_tokenizer=True
#         )
#         decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#         label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

#         sources += [source.strip() for source in decoded_sources]
#         preds += [pred.strip() for pred in decoded_preds]
#         labels += [[label.strip()] for label in decoded_labels]
#     bleu_score = bleu.corpus_score(preds, labels).score
#     print(f"Test BLEU: {bleu_score:>0.2f}\n")
#     results = []
#     print('saving predicted results...')
#     for source, pred, label in zip(sources, preds, labels):
#         results.append({
#             "sentence": source, 
#             "prediction": pred, 
#             "translation": label[0]
#         })
#     with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
#         for exapmle_result in results:
#             f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
