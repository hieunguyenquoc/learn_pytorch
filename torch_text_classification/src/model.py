import torch
import torch.nn as nn
from preprocess_data import data_classication

class Text_classification(nn.Module):
    def __init__(self):
        super(Text_classification,self).__init__()
        #parameter
        self.data_preprocess = data_classication()
        self.data_preprocess.tokenization()
        self.num_embedding = len(self.data_preprocess.tokens)
        self.embedding_dim = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.num_class = 18
        
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
        
        out = torch.relu_(out[:,-1,:]) #chọn ra bước thời gian cuối cùng của mỗi chuỗi trong batch. Dấu : trước -1 có nghĩa là chọn tất cả các phần tử theo chiều đầu tiên (kích thước batch), và sau -1 có nghĩa là chọn bước thời gian cuối cùng theo chiều thứ hai (độ dài chuỗi).
        
        out = self.linear(out)
        
        return out
        
        