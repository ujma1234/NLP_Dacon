import torch
from torch import nn

device = torch.device("cuda:0")
class BERT(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERT, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float())
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return out

class RoBERTa(nn.Module):
    def __init__(self,
                bert,
                hidden_size = 768,
                num_classes=2,
                dr_rate=None,
                params=None):
        super(RoBERTa, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, token_type_ids, attention_mask):
        pooler = self.bert(input_ids, token_type_ids, attention_mask)['pooler_output']
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return out