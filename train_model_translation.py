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

import argparse
import transformers
import pdb
import torch.nn as nn

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

# invalid implementation
class MarianMTModelFinetuned(nn.Module):
    def __init__(self, checkpoint):
        super(MarianMTModelFinetuned, self).__init__()
        self.marian_mt_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.decoder_vocab_size = self.marian_mt_model.config.decoder_vocab_size
        print(self.decoder_vocab_size)
        self.translator = [nn.ReLU(), nn.Linear(self.decoder_vocab_size, self.decoder_vocab_size)]
        self.translator = nn.Sequential(*self.translator)
    
    def forward(self, input_ids, attention_mask, decoder_input_ids, labels):
        lm_logits = self.marian_mt_model(input_ids, attention_mask, decoder_input_ids, labels).logits
        # pdb.set_trace()
        lm_logits = self.translator(lm_logits)
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.decoder_vocab_size), labels.view(-1))
        return transformers.modeling_outputs.Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits)

def collote_fn(batch_samples, model, tokenizer, options):
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
        if options.full_tuning and options.base_model_ckpt == 'Helsinki-NLP/opus-mt-zh-en':
            batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        elif options.base_model_ckpt == 'Helsinki-NLP/opus-mt-zh-en':
            batch_data['decoder_input_ids'] = model.marian_mt_model.prepare_decoder_input_ids_from_labels(labels)
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
        outputs = model(**batch_data) # outputs.logits.shape=[b, 51, num_token_classes]=[b, 51, 65001]
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

def test_loop(options, dataloader, model, mode='test'):
    assert mode in ['valid', 'test']
    preds, labels = [], []

    if mode == 'test':
        if options.full_tuning:
            model.load_state_dict(torch.load('%s/%s_finetuned_weights.pt'%(options.save_path, options.base_model_ckpt.replace('/', '-'))))
        else:
            model.translator.load_state_dict(torch.load('%s/%s_head_finetuned_weights.pt'%(options.save_path, options.base_model_ckpt.replace('/', '-'))))
    
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(device)
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
    print(f"BLEU: {bleu_score:>0.5f}\n")
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
    parser.add_argument('--full_tuning', type=bool, default=False, help='train all parameters or only train the head')
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

    model_checkpoint = options.base_model_ckpt
    if options.base_model_ckpt == 'Helsinki-NLP/opus-mt-zh-en':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        '''
        MarianTokenizer(name_or_path='Helsinki-NLP/opus-mt-zh-en', vocab_size=65001, model_max_length=512, is_fast=False, padding_side='right', 
        truncation_side='right', special_tokens={'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=True)
        '''
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        if options.full_tuning:
            model = transformers.MarianMTModel.from_pretrained(model_checkpoint)
        else:
            raise Exception('[INFO] Under construction')
            model = MarianMTModelFinetuned(model_checkpoint)
    elif options.base_model_ckpt == 'facebook/nllb-200-distilled-600M':
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        # model = AutoModel.from_pretrained(model_checkpoint)
    else:
        raise Exception('[INFO] Invalid base model checkpoint')
    # print(tokenizer)
    # print(model)
    model = model.to(device)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collote_fn(x, model, tokenizer, options))
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collote_fn(x, model, tokenizer, options))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collote_fn(x, model, tokenizer, options))

    bleu = BLEU()

    if options.full_tuning:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.translator.parameters(), lr=learning_rate)
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
        valid_bleu = test_loop(options, valid_dataloader, model, mode='valid')
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            print('saving new weights...\n')
            if options.full_tuning:
                torch.save(model.state_dict(), '%s/%s_finetuned_weights.pt'%(options.save_path, options.base_model_ckpt.replace('/', '-')))
            else:
                torch.save(model.translator.state_dict(), '%s/%s_head_finetuned_weights.pt'%(options.save_path, options.base_model_ckpt.replace('/', '-')))
            with open('%s/log.txt'%options.save_path, 'w') as w:
                w.write('best checkpoint epoch: %d\n'%(t + 1))
                w.write('best checkpoint acc: %.05f'%best_bleu)

    test_loop(options, dataloader, model, mode='test')
    
    '''
    MarianMTModel(
    (model): MarianModel(
        (shared): Embedding(65001, 512, padding_idx=65000)
        (encoder): MarianEncoder(
        (embed_tokens): Embedding(65001, 512, padding_idx=65000)
        (embed_positions): MarianSinusoidalPositionalEmbedding(512, 512)
        (layers): ModuleList(
            (0): MarianEncoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (activation_fn): SiLUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (1): MarianEncoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (activation_fn): SiLUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (2): MarianEncoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (activation_fn): SiLUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (3): MarianEncoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (activation_fn): SiLUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (4): MarianEncoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (activation_fn): SiLUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (5): MarianEncoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (activation_fn): SiLUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
        )
        )
        (decoder): MarianDecoder(
        (embed_tokens): Embedding(65001, 512, padding_idx=65000)
        (embed_positions): MarianSinusoidalPositionalEmbedding(512, 512)
        (layers): ModuleList(
            (0): MarianDecoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (activation_fn): SiLUActivation()
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (encoder_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (1): MarianDecoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (activation_fn): SiLUActivation()
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (encoder_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (2): MarianDecoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (activation_fn): SiLUActivation()
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (encoder_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (3): MarianDecoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (activation_fn): SiLUActivation()
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (encoder_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (4): MarianDecoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (activation_fn): SiLUActivation()
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (encoder_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
            (5): MarianDecoderLayer(
            (self_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (activation_fn): SiLUActivation()
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (encoder_attn): MarianAttention(
                (k_proj): Linear(in_features=512, out_features=512, bias=True)
                (v_proj): Linear(in_features=512, out_features=512, bias=True)
                (q_proj): Linear(in_features=512, out_features=512, bias=True)
                (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            )
        )
        )
    )
    (lm_head): Linear(in_features=512, out_features=65001, bias=False)
    )
    '''

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
