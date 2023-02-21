import torch
import transformers
from torch import nn
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import csv
import gluonnlp as nlp
from tqdm import tqdm

device = torch.device("cuda:0")

tokenizer = transformers.AutoTokenizer.from_pretrained("klue/bert-base")

dataset_train = []
dataset_test = []

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

    #testtest

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

# dataset_train = make_data_list(dataset_train)
dataset_test = make_data_list(dataset_test)

dataset_test = dataset_test[1:]

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

# data_train = RoBERTaDataset(dataset_train, len(dataset_train), tokenizer, 64)
data_test = RoBERTaDataset(dataset_test, len(dataset_test), tokenizer, 64)

# train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=1, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, num_workers=0)


class SentenceClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained('klue/bert-base')
        self.lstm = nn.LSTM(input_size=768, hidden_size=1024, num_layers=1, bidirectional=True, batch_first = True)
        self.fc = nn.Linear(2048, num_classes)
        self.drop = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            PAD_pooler = torch.zeros(1, 768, dtype = torch.float32).to(device)
            pooler = []
            for _ in range(64 - len(x)):
                pooler += PAD_pooler.tolist()
            for input_ids, attention_mask, segment_ids in x:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                segment_ids = segment_ids.to(device)
                pooler_bert = self.bert(input_ids, attention_mask, segment_ids)
                # pooler_bert = self.drop(pooler_bert[1])
                pooler_bert = pooler_bert[1]
                pooler += pooler_bert.tolist()

            pooler = torch.tensor(pooler, dtype=torch.float32).to(device)
            pooler = pooler.reshape(1, 64, 768).to(device)

        
        # Extract the last hidden state of the BERT model
        # _, last_hidden_state = self.bert(input_ids, attention_mask=attention_mask)
        
        # Pass the last hidden state through the LSTM
        lstm_out, (hn, cn) = self.lstm(pooler)

        hn = hn.view(-1)
        hn = hn.reshape(1, 2048)

        hn = self.relu(hn)
        # hn = self.drop(hn)

        # Average the LSTM outputs for each direction
        # lstm_out = lstm_out.mean(dim=0)
        
        # Pass the LSTM output through a fully connected layer to make the final prediction
        logits = self.fc(hn)

        logits = logits.reshape(1, 3)
        return logits
    
# ckpt = torch.load(".cache/robertNlstm-68.pt", map_location=device)

# Define the model
model = SentenceClassificationModel(num_classes=3).to(device)
# model.load_state_dict(ckpt['model_state_dict'])

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=(2e-5)*2)
# optimizer.load_state_dict(ckpt['optimizer_state_dict'])

def calc_accuracy(X,Y):
    _, index = torch.max(X, 1)
    if(index == Y):
        train_acc = 1
    else:
        train_acc = 0
    return train_acc

checkpoint = 58

for epoch in range(11):
    ckpt = torch.load(f".cache/robertNlstm-{checkpoint+epoch}.pt", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with open(f"{data_path}submission-{epoch}.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(['index','category'])
        for i ,(x, labels) in enumerate(tqdm(test_dataloader, total = len(test_dataloader))):
            with torch.no_grad():
                logits = model(x)
            _, index = torch.max(logits, 1)
            writer.writerow([int(labels), int(index)])
        file.close()

# # Define the training loop
# for epoch in range(10):
#     train_acc = 0.0
#     total_loss = 0
#     for i ,(x, labels) in enumerate(tqdm(train_dataloader, total = len(train_dataloader))):
#         # Zero out the gradients from any previous steps
#         optimizer.zero_grad()

#         # Make predictions using the model
#         logits = model(x)

#         labels = labels[0].reshape(1)

#         # Calculate the loss between the true labels and the predictions
#         loss = criterion(logits, labels)

#         train_acc += calc_accuracy(logits, labels)

#         # Perform backpropagation to calculate the gradients
#         loss.backward()

#         # Update the model parameters
#         optimizer.step()

#         total_loss += loss.item()
        
#     # Print the average loss for the epoch
#     print(f"Epoch {epoch}: Loss = {total_loss / len(train_dataloader)} Train_acc = {train_acc / len(train_dataloader)}")

#     torch.save(
#             {
#                 "model":"RoBERTa-LSTM",
#                 "epoch":epoch,
#                 "model_state_dict":model.state_dict(),
#                 "optimizer_state_dict":optimizer.state_dict(),
#                 "description":f"RoBERTa-LSTM 체크포인트-{checkpoint}",
#             },
#             f".cache/robertNlstm-{checkpoint}.pt",
#         )
#     checkpoint += 1