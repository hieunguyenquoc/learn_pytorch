from torch.utils.data import Dataset, DataLoader

class data(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.label[index]


        