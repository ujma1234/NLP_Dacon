import torch
from torch import nn

from my_module import ps_bert
from my_module import ps_gru

device = "cuda" if torch.cuda.is_available() else "cpu"

class GRUBERT(nn.Module):
    def __init__(self, hidden_size, fc_size, num_layers, bertmodel,dr_rate, bert_type = 0):
        super(GRUBERT, self).__init__()
        self.bert_type = bert_type
        if self.bert_type == 0:
            self.bert = ps_bert.BERT(bertmodel, dr_rate[0])
        elif self.bert_type == 1:
            self.bert = ps_bert.RoBERTa(bertmodel, dr_rate[0])
        self.gru = ps_gru.GRU(num_classes = 1, input_size = 768, hidden_size = hidden_size, num_layers = num_layers, seq_length = 768, dr_rate = dr_rate[1])
        self.num_classes = 4
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.dr_rate = dr_rate[2]

        if self.dr_rate:
            self.dropout = nn.Dropout(p=dr_rate[2])

        self.month_classifier = nn.Linear(self.fc_size, 12)
        self.month_fc = nn.Linear(hidden_size, self.fc_size)
        self.day_classifier = nn.Linear(self.fc_size, 31)
        self.day_fc = nn.Linear(hidden_size, self.fc_size)
        self.hour_classifier = nn.Linear(self.fc_size, 24)
        self.hour_fc = nn.Linear(hidden_size, self.fc_size)
        self.min_classifier = nn.Linear(self.fc_size, 12)
        self.min_fc = nn.Linear(hidden_size, self.fc_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        PAD_pooler = torch.zeros(1, 768, dtype = torch.float32)
        schedule_out = []
        pooler = []
        for _ in range(64 - len(x)):
            pooler += PAD_pooler.tolist()

        for token_ids, valid_length, segment_ids in x:
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length.to(device)
            pooler += self.bert(token_ids, valid_length, segment_ids).tolist()

        pooler = torch.tensor(pooler, dtype=torch.float32).to(device)
        pooler = pooler.reshape(1, 64, 768)

        seq_len = torch.tensor(len(x)).to(device)

        out = self.gru(pooler, seq_len).to(device)
        if self.dr_rate:
            out = self.dropout(out)

        m_out = self.month_fc(out)
        m_out = self.relu(m_out)
        d_out = self.day_fc(out)
        d_out = self.relu(d_out)
        h_out = self.hour_fc(out)
        h_out = self.relu(h_out)
        mi_out = self.min_fc(out)
        mi_out = self.relu(mi_out)

        if self.dr_rate:
            m_out = self.dropout(m_out)
            d_out = self.dropout(d_out)
            h_out = self.dropout(h_out)
            mi_out = self.dropout(mi_out)

        m_out = self.month_classifier(m_out)
        d_out = self.day_classifier(d_out)
        h_out = self.hour_classifier(h_out)
        mi_out = self.min_classifier(mi_out)

        schedule_out = [m_out, d_out, h_out, mi_out]
        
        return schedule_out