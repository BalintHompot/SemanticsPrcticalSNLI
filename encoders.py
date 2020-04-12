import torch
from torch import nn
from torch import functional as F

class EncoderModel(nn.Module):
    def __init__(self, params):
        super(EncoderModel, self).__init__()
        self.name = "Parent encoder model"
        self.layers = None
        self.input_size = 300
        self.dropout = 0.5
        self.embedding = nn.Embedding(params["vocab_size"], params["vector_size"], padding_idx= params["pad_idx"]).from_pretrained(params["pretrained"])
    
    def processPadded(self, textTuple):
        text, text_lens = textTuple
        self.lens = text_lens
        embedded = self.embedding(text)
        return torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lens, batch_first=True, enforce_sorted=False)

    def forward(self, x):
        x = self.processPadded(x)
        out, _ = self.layers(x)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)
        ## get the last hidden
        idx = (self.lens - 1).view(-1, 1).expand(len(self.lens), unpacked.size(2))
        idx = idx.unsqueeze(1)
        last_hidden = torch.gather(unpacked, dim = 1, index = idx)
        return torch.squeeze(last_hidden)

class MeanEncoder(EncoderModel):
    def __init__(self, params, hidden_size, layer_num):
        super(MeanEncoder, self).__init__(params)
        self.name = "Vector mean"
        self.layer_size = self.input_size
        
    def forward(self, x):
        text, text_lens = x
        text_lens = text_lens.float()
        embedded = self.embedding(text)
        averaged = torch.sum(embedded, dim = 1)
        averaged = torch.div(averaged, text_lens.view(-1,1))
        return averaged

class LSTMEncoder(EncoderModel):
    def __init__(self, params, hidden_size, layer_num):
        super(LSTMEncoder, self).__init__(params)
        self.name = "LSTM"
        self.layer_size = hidden_size
        self.layers = nn.LSTM(self.input_size, hidden_size, num_layers=layer_num, dropout=self.dropout, bidirectional=False, batch_first=True)

class BiLSTMEncoder(EncoderModel):
    def __init__(self, params, hidden_size, layer_num):
        super(BiLSTMEncoder, self).__init__(params)
        self.name = "BiLSTM"
        self.layer_size = hidden_size *2
        self.layers = nn.LSTM(self.input_size, hidden_size, num_layers=layer_num, dropout=self.dropout, bidirectional=True, batch_first=True)


class MaxBiLSTMEncoder(EncoderModel):
    def __init__(self, params, hidden_size, layer_num):
        super(MaxBiLSTMEncoder, self).__init__(params)
        self.name = "Pooled BiLSTM"
        self.layer_size = hidden_size *2
        self.layers = nn.LSTM(self.input_size, hidden_size, num_layers=layer_num, dropout=self.dropout, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        x = self.processPadded(x)
        out, _ = self.layers(x)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first = True)
        max_pooled = torch.squeeze(torch.max(unpacked, dim = 1).values)

        return max_pooled
