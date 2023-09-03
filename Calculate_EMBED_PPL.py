import numpy as np
import pandas as pd
import json
from transformers import LongformerTokenizer, LongformerModel, LongformerConfig, AutoTokenizer, LongformerForMaskedLM, \
    AutoModelForMaskedLM, AutoModel, BertTokenizer, BertForMaskedLM, BertConfig
import torch
import math

n_token = 0


def Forward(model, tokenizer, text: str)->np.array:
    """
    Calculate the embeddings of the given text using the last hidden layer of the model
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokens = tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt")
    tokens = tokens.to(device)
    model = model.to(device)
    with torch.no_grad():
        output = model(tokens['input_ids'], attention_mask=tokens['attention_mask'], return_dict=True,
                       output_hidden_states=True)
        token_embedding = output.hidden_states[-1]
        sentence_embedding = torch.mean(token_embedding, 1)
        out_np = sentence_embedding.detach().numpy()
        return out_np


def Perplexity(model, tokenizer, text: str)->float:
    """
    Compute the loss of the model on a given text (perplexity is equal to math.exp(loss/n_tokens)
    As the model has a fixed input size, the model can process a text with sliding  window approach where the sequence
    is broken into the subsequences equal to the model’s maximum input size and the final loss is the mean of the losses
    of the subsequences
    """
    global n_token
    tokens = tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt")
    n_token += len(tokens['input_ids'])
    max_length, stride, seq_len = 512, 512, tokens.input_ids.size(1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    nlls = []
    prev_end_loc = 0
    for i in range(0, seq_len, stride):
        end_loc = min(i + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = tokens.input_ids[:, i:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Masked the tokens
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood.item())
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return np.mean(nlls)


# Upload Tokenizer and Model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_path = "<path>"
config = LongformerConfig.from_json_file(f'{model_path}/config.json')
# config = BertConfig.from_json_file(f'{model_path}/config.json')
# model = LongformerForMaskedLM.from_pretrained(f'{model_path}/pytorch_model.bin', config=config)
model = AutoModelForMaskedLM.from_pretrained(f'{model_path}/pytorch_model.bin', config=config)
# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
# attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local
# attention
# attention_mask[:, [...]] =  2  # Set global attention based on the task. For example,
# classification: the <s> token
# QA: question tokens
# global_attention_mask for global as well

df = pd.read_json("<path>/AmazonCat-13K/tst.json", lines=True)
vec, Loss = [], []
pp = False
for i, j in enumerate(df['content']):
    output = Forward(model, tokenizer, j)
    vec.append(output)
    loss = Perplexity(model, tokenizer, j)
    Loss.append(loss)
ppl = math.exp(sum(Loss) / n_token)
print(ppl, sum(Loss) / n_token)
np.savetxt(fname="<path>/embeddings.txt", X=vec)


