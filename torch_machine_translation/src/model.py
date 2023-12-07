import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        energy = F.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, attention):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = attention

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        attention_weights = self.attention(hidden[-1], encoder_outputs)
        context = attention_weights.unsqueeze(0).unsqueeze(0).bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden