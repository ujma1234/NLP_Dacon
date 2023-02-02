import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda:0")

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dr_rate):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_lenght = seq_length
        self.dr_rate = dr_rate

        if self.dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 2048)
        self.fc = nn.Linear(2048, hidden_size)

        self.relu = nn.ReLU()
    
    def forward(self,x, seq_num):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1)
        out = self.relu(hn)
        if self.dr_rate:
            out = self.dropout(out)
        out = self.fc_1(out) 
        out = self.relu(out)
        if self.dr_rate:
            out = self.dropout(out)
        out = self.fc(out)
        return out