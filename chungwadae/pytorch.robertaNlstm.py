# KLUE-RoBERTa-base
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
import os.path
import gluonnlp as nlp
from tqdm.notebook import tqdm
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import csv
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_module import ps_bertNlstm

device = torch.device("cuda:0")

bertmodel = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
file_num = 0

dataset_train = []
dataset_test = []

import numpy as np

data_path = "/home/nam/Documents/data/"
with open(data_path+"train.csv", 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        dataset_train.append(row[1:])

with open(data_path+"test.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset_test.append(row)

def make_data_list(dataset):
    cnt = 0
    check = 0
    for j in (dataset[k][1] for k in range(len(dataset))):
        tmp = []
        tmp_list = []
        for i in j:
            if i == "\\":
                check = 1
                tmp = ''.join(tmp)
                if(len(tmp) > 128):
                    l = tmp.split(".")
                    for t in l:
                        if not tmp_list == []:
                            if t == '':
                                ll = tmp_list.pop(-1) 
                                ll[0]+="."
                                tmp_list.append(ll)
                            else:
                                tmp_list.append([t])
                        else:
                            if t == '':
                                continue
                            else :
                                tmp_list.append([t])
                else:
                    tmp_list.append([tmp])
                tmp = []
            elif check and i == "n":
                    check = 0
                    continue
            else:
                tmp += i
        if tmp_list == []:
            tmp = dataset[cnt][1].split(".")
            for i in tmp:
                if i == '':
                    if tmp_list != []:
                        l = tmp_list.pop(-1)
                        l[0] += "."
                        tmp_list.append(l)
                    else:
                        continue
                else:
                    tmp_list.append([i])
        dataset[cnt].pop(1)
        dataset[cnt].append(tmp_list)
        cnt += 1
    return dataset

dataset_train = make_data_list(dataset_train)
dataset_test = make_data_list(dataset_test)

# print(dataset_train)

class RoBERTaDataset(Dataset):
    def __init__(self, dataset, data_num, tokenizer, max_len):
        # data = [j for j in (dataset_train[i][1:] for i in range(train_num))]
        self.sentences = []
        for i in range(data_num):
            temp = []
            for j in dataset[i][1]:
                tmp = []
                encoded_dict = tokenizer(
                    text = j[0],
                    add_special_tokens = True,
                    max_length = max_len,
                    pad_to_max_length = True,
                    truncation = True,
                    return_tensors="pt"
                ).to(device)
                tmp.extend(encoded_dict['input_ids'])
                tmp.extend(encoded_dict['token_type_ids'])
                tmp.extend(encoded_dict['attention_mask'])
                temp += [tmp]
                if(len(temp) >= 64):
                    break
            self.sentences.append(temp)

        # Make labels 
        self.labels = []
        for i in range(data_num):
            classify = int(dataset[i][0])

            label_list = torch.tensor([classify]).to(device)

            self.labels += [label_list]

    def __getitem__(self, i):
        return ([self.sentences[i]] + [(self.labels[i])])

    def __len__(self):
        return (len(self.labels))

dr_rates = [0.3, 0.3, 0.3]
max_len = 64
batch_size = 1
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

data_train = RoBERTaDataset(dataset_train, len(dataset_train), tokenizer, max_len)
# data_test = RoBERTaDataset(dataset_test, len(dataset_test), tokenizer, max_len)

# print(data_train.__getitem__(1))

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
# test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)
print(len(train_dataloader))
bertmodel = bertmodel.to(device)
        
model = ps_bertNlstm.LSBERT(hidden_size = 768, fc_size = 2048, num_layers=1, bertmodel = bertmodel, dr_rate = dr_rates, bert_type=1).to(device)

# checkpoint = torch.load(".cache/robertNlstm-1.pt", map_location=device)
# # model.load_state_dict(checkpoint['model_state_dict'])

# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in model.parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# # n=0
# # for name, child in model.named_children():
# #     if n==0:
# #       h=0
# #       for param in child.parameters():
# #         if h<=328: #이부분 숫자 조절로 fine-tuning => Roberta229: h=229
# #           param.requires_grad = False
# #         h+=1
# #     n+=1

optimizer = AdamW(model.parameters(), lr=learning_rate)
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# # print(checkpoint["description"])

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def make_loss_N_Backward(data, label):
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    target = label[0].reshape(1)
    input = data[0].reshape(1,3)
    # print(input)
    loss = loss_fn(input, target)
    loss.backward()
    losses.append(loss.tolist())
    return losses

def calc_accuracy(X,Y):
    pred = []
    vals = []
    indices = []
    mon = X[0].reshape(1, 3)
    pred = [mon]
    Y = Y[0]
    for i, j in (torch.max(k, 1) for k in pred):
        vals += i
        indices += j
        # print(j)
    indices = torch.tensor(indices).to(device)
    train_acc = (indices == Y).sum().data.cpu().numpy()/indices.size()[0]
    return train_acc
    
checkpoint = 1
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (x, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        predict = []
        out = model(x)
        # print(label)
        loss = make_loss_N_Backward(out, label)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss, train_acc / (batch_id+1)))
            checkpoint += 1
        # print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    torch.save(
                {
                    "model":"RoBERTa-LSTM",
                    "epoch":e,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "description":f"RoBERTa-LSTM 체크포인트-{checkpoint}",
                },
                f".cache/robertNlstm-{checkpoint}.pt",
            )
            






    # model.eval()
# #     for batch_id, (x, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
# #         label = label.long().to(device)
# #         out = model(x)
# #         test_acc += calc_accuracy(out, label)
# #     print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        

# # Training and Evaluate
# checkpoint = 1
# model.train()
# for epoch in range(num_epochs):
#     cost = 0.0

#     for batch_id, (x, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
#         predict = []
#         out = model(x)
#         loss = make_loss_N_Backward(out, label)
#         optimizer.step()

#         cost += calc_accuracy(out, label)

#     print("epoch {} train acc  = {} ({} / {}+1)".format(epoch, cost / (batch_id+1), cost, batch_id))
    
#     cost = cost / len(train_dataloader)

#     if (epoch + 1) % 5 == 0:
#         torch.save(
#             {
#                 "model":"RoBERTa-LSTM",
#                 "epoch":epoch,
#                 "model_state_dict":model.state_dict(),
#                 "optimizer_state_dict":optimizer.state_dict(),
#                 "cost":cost,
#                 "description":f"RoBERTa-LSTM 체크포인트-{checkpoint}",
#             },
#             f".cache/robertNlstm-{checkpoint}.pt",
#         )
#         checkpoint += 1