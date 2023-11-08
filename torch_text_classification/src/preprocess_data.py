from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re

class data_classication:
    def __init__(self):
        self.path_data = "torch_text_classification/data/news_categories.txt"
        self.test_size = 0.2

    def clean_data(self, text):
        text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
        text = text.lower()
        return text
    
    def load_data(self):
        f = open(self.path_data, "r", encoding="utf-8")
        label = []
        self.text = []

        for i in f:
            label.append(i.split()[0])
            self.text.append(" ".join(i.split()[1:]))
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.text, label, test_size=self.test_size, shuffle=True)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.Y_train)

        self.Y_train = self.label_encoder.transform(self.Y_train)
        self.Y_test = self.label_encoder.transform(self.Y_test)
    
    def tokenization(self):
        self.tokens = [word for sentence in self.X_train for word in word_tokenize(self.clean_data(sentence))]
        self.tokens = {word: self.tokens.count(word) for word in self.tokens}
        
    def sequence_to_token(self, input):
        idx = []
        for i in input:
            idx_s = []
            input_token = word_tokenize(i.lower())
            for token in input_token:
                if token in self.tokens:
                    idx_s.append(list(self.tokens.keys()).index(token))
                else:
                    idx_s.append(0)
            idx.append(idx_s)
        return pad_sequences(idx, maxlen=20)


    # def create_vocab(self):

        


               