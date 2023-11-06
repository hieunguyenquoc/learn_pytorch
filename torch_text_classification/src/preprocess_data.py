from sklearn.model_selection import train_test_split

class data_classication():
    def __init__(self):
        self.path_data = "torch_text_classification/data/news_categories.txt"
        self.test_size = 0.2

    def load_data(self):
        f = open(self.path_data, "r", encoding="utf-8")
        label = []
        self.text = []

        for i in f:
            label.append(i.split()[0])
            self.text.append(" ".join(i.split()[1:]))
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(text, label, test_size=self.test_size, shuffle=True)

    # def create_vocab(self):

        


               