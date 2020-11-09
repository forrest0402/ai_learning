# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 10/11/2020 上午1:37
"""

import torch
import tqdm
from nlp import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = 'cuda'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
# encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

print('load completely')


def ppl(text):
    max_length = model.config.n_positions
    stride = 512
    encodings = tokenizer(text, return_tensors='pt')
    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / (i + 1))
    return ppl


texts = ['I want to go to Beijing',
         'Beijing I want to go to you',
         'This is really a long weekend',
         'This weekend is long really',
         '我要去北京', '北京要去我', '这真的是一个漫长的周末啊']

for text in texts:
    print(f'ppl: {ppl(text)} for {text}')

print(ppl('\n\n'.join(test['text'])))
