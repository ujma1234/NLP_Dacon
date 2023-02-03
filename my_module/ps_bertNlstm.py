import torch
from torch import nn

from my_module import ps_lstm
from my_module import ps_bert

device = torch.device("cuda:0")

class LSBERT(nn.Module):
    ## hidden_size = 전달받는 은닉층의 크기, fc_size = 신경망 크기, num_layers = lstm_sell 크기
    def __init__(self, hidden_size, fc_size, num_layers, bertmodel, dr_rate, bert_type = 0):
        super(LSBERT, self).__init__()
        self.bert_type = bert_type
        if self.bert_type == 0:
            self.bert = ps_bert.BERT(bertmodel, dr_rate=dr_rate[0])
        elif self.bert_type == 1:
            self.bert = ps_bert.RoBERTa(bertmodel, dr_rate=dr_rate[0])
        self.f_lstm = ps_lstm.LSTM(num_classes = 1, input_size = 768, hidden_size = hidden_size, num_layers = num_layers, seq_length = 768, dr_rate = dr_rate[1])
        self.num_classes = 4
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.dr_rate = dr_rate[2]

        if self.dr_rate:
            self.dropout = nn.Dropout(p=dr_rate[2])

        self.out_classifier = nn.Linear(self.fc_size, 3)
        self.out_fc = nn.Linear(hidden_size, self.fc_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        PAD_pooler = torch.zeros(1, 768, dtype = torch.float32)
        pooler = []
        for _ in range(64 - len(x)):
            pooler += PAD_pooler.tolist()
            
        for token_ids, valid_length, segment_ids in x:
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length.to(device)
            pooler += self.bert(token_ids, valid_length, segment_ids).tolist()

        # print(pooler)
        pooler = torch.tensor(pooler, dtype=torch.float32).to(device)
        # print(pooler.size())
        pooler = pooler.reshape(1, 64, 768)

        seq_len = torch.tensor(len(x)).to(device)

        out = self.f_lstm(pooler, seq_len).to(device)
        # print(out.size())
        if self.dr_rate:
            out = self.dropout(out)

        m_out = self.out_fc(out)
        m_out = self.relu(m_out)

        if self.dr_rate:
            m_out = self.dropout(m_out)

        m_out = self.out_classifier(m_out)
        
        return [m_out]