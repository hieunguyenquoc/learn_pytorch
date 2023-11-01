import torch
import torch.nn as nn

class Text_classification(nn.Module):
    def __init__(self, num_embedding, embedding_dim, num_layer, dropout_rate, num_class):
        super(Text_classification,self).__init__()
        #parameter
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.num_layer = num_layer
        self.dropout = dropout_rate
        self.num_class = num_class
        
        #network stucture
        self.embedd = nn.Embedding(num_embeddings=self.num_embedding, embedding_dim=self.embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size = self.embedding_dim, hidden_size = self.embedding_dim, num_layers = self.num_layer, batch_first = True)
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(in_features=self.embedding_dim, out_features=18)
        
    def forward(self, x):
        out = self.embedd(x)
        
        h = torch.zeros((self.num_layer, x.size(0), self.embedding_dim))
        
        out, h = self.rnn(out, h)
        
        out = self.dropout(out)
        
        out = torch.relu_(out[:,-1,:])
        
        out = self.linear(out)
        
        return out
        
        