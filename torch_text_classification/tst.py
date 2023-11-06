from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from underthesea import word_tokenize
import re

class Preprocessing:
    def tokenization(self, X_train):
        self.tokens = word_tokenize(X_train)
        return self.tokens

def clean_data(text):
  text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
  text = text.lower()
  return text

X_train = [
    "Đây là câu số 1.",
    "Đây là câu số 2.",
    "Câu số 3 khác với câu số 1.",
]

preprocess = Preprocessing()
X_tokens = [word for sentence in X_train for word in word_tokenize(clean_data(sentence))]
X_tokens = {word: X_tokens.count(word) for word in X_tokens}
print(X_tokens)

input = "Đây là câu số 2"
input_token = word_tokenize(input.lower())
idx = []
for token in input_token:
    if token in X_tokens:
        idx.append(list(X_tokens.keys()).index(token))
    else:
        idx.append(0)

print(idx)





