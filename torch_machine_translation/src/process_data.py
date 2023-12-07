import pandas as pd
from underthesea import word_tokenize
from collections import Counter
import string

class process_data_for_train:
    
    def read_csv_file(self):
        # Đọc dữ liệu từ file CSV
        self.data = pd.read_csv('data_csv/train.csv', nrows=500)

        # Loại bỏ các dòng có giá trị null
        self.data = self.data.dropna()
        
        #Loại bỏ \n
        self.data['en'] = self.data['en'].str.rstrip('\n')
        self.data['vi'] = self.data['vi'].str.rstrip('\n')
        
        return self.data
    
    @staticmethod
    def tokenize_en(text):
        #token tiếng Anh
        return text.split(" ")
    
    @staticmethod
    def tokenize_vi(text):
        #token tiếng Việt
        return word_tokenize(text)
    
    @staticmethod
    def add_special_token(text):
        #thêm ký tự đặc biệt
        return ['<sos>'] + text + ['<eos>']

    @staticmethod
    def create_en_vocab(data):
        en_vocab = Counter([word for tokens in data['en_tokens'] for word in tokens])
        en_vocab_special = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}  # Thêm token đặc biệt
        en_vocab_special.update({word: idx for idx, word in enumerate(en_vocab.keys(), start=4)})
        return en_vocab_special
    
    @staticmethod
    def create_vi_vocab(data):
        vi_vocab = Counter([word for tokens in data['vi_tokens'] for word in tokens])
        vi_vocab_special = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}  # Thêm token đặc biệt
        vi_vocab_special.update({word: idx for idx, word in enumerate(vi_vocab.keys(), start=4)})
        return vi_vocab_special
    
    @staticmethod
    def remove_punctuation(text):
    # Loại bỏ các ký tự dấu câu
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)

        return text
            
# if __name__ == "__main__":
#     df = process_data_for_train()
#     train_data = df.read_csv_file()
#     tokenize_en = df.tokenize_en
#     tokenize_vi = df.tokenize_vi
#     add_special_token = df.add_special_token
#     create_en_vocab = df.create_en_vocab
#     create_vi_vocab =df.create_vi_vocab
#     remove_punctual = df.remove_punctuation
    
#     train_data["en"] = train_data["en"].apply(remove_punctual)
#     train_data["vi"] = train_data["vi"].apply(remove_punctual)
    
#     train_data['en_tokens'] = train_data['en'].apply(tokenize_en)
#     train_data['vi_tokens'] = train_data['vi'].apply(tokenize_vi)
    
#     train_data['en_tokens'] = train_data['en_tokens'].apply(add_special_token)
#     train_data['vi_tokens'] = train_data['vi_tokens'].apply(add_special_token)
    
#     en_vocab = create_en_vocab(train_data)
#     vi_vocab = create_vi_vocab(train_data)
    
#     print(en_vocab)
#     # print(train_data.head)