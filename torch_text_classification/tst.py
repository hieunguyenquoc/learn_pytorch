from keras.utils import pad_sequences
from underthesea import word_tokenize
import re

class Preprocessing:
    def clean_data(self, text):
        text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
        text = text.lower()
        return text
    
    def tokenization(self, X_train):
        self.tokens = [word for sentence in X_train for word in word_tokenize(self.clean_data(sentence))]
        self.tokens = {word: self.tokens.count(word) for word in self.tokens}

    def sequence_to_vecto(self, input):
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



X_train = [
    "Đây là câu số 1.",
    "Đây là câu số 2.",
    "Câu số 3 khác với câu số 1.",
]

input = [
    "Đây là câu số 2.",
    "Đây là câu số 4.",
    "Câu số 3 khác với câu số 2.",
]

preprocess = Preprocessing()
preprocess.tokenization(X_train)
print(preprocess.sequence_to_vecto(input))
# X_tokens = [word for sentence in X_train for word in word_tokenize(clean_data(sentence))]
# X_tokens = {word: X_tokens.count(word) for word in X_tokens}
# print(X_tokens)

# input = "Đây là câu số 2"
# input_token = word_tokenize(input.lower())
# idx = []
# for token in input_token:
#     if token in X_tokens:
#         idx.append(list(X_tokens.keys()).index(token))
#     else:
#         idx.append(0)






