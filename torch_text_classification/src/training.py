import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocess_data import data_classication
from src.model import Text_classification
import torch.optim as optim
import torch.nn as nn

class customDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Trainer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.preprocess = data_classication()
        self.preprocess.tokenization()

        #load raw data
        raw_x_train = self.preprocess.X_train
        raw_x_test = self.preprocess.X_test

        self.x_train = self.preprocess.sequence_to_token(raw_x_train)
        self.x_test = self.preprocess.sequence_to_token(raw_x_test)

        self.y_train = self.preprocess.Y_train
        self.y_test = self.preprocess.Y_test

        #load model
        self.model = Text_classification()
        self.model.to(self.device)

        #hyperameter
        self.learning_rate = 0.01
        self.batch_size = 64
        self.epochs = 5

    def train(self):
        #create Dataset
        train_data = customDataset(self.x_train, self.y_train)
        test_data = customDataset(self.x_test, self.y_test)

        #Create DataLoader
        self.train_iter = DataLoader(train_data, batch_size=self.batch_size)
        self.test_iter = DataLoader(test_data)

        #Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        #loss function
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        loss_fn = loss_fn.to(self.device)
        #Mode train
        self.model.train()
        print(self.model)
        
        for epoch in range(self.epochs):
            prediction = []
            for x_batch, y_batch in self.train_iter:
                #pass the data in iterable
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.LongTensor)

                x = x.to(self.device)
                y = y.to(self.device)

                #get prediction in training phase
                y_pred = self.model(x)

                #get loss value
                loss = loss_fn(y_pred, y)

                #optimizer
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                prediction += list(y_pred.cpu().squeeze().detach().numpy())

        torch.save(self.model.state_dict(),"model/model.pt")






        