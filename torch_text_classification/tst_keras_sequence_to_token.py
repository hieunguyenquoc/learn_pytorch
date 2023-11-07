from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tạo một đối tượng Tokenizer và đánh chỉ mục từ vựng
tokenizer = Tokenizer()
text_data = [
    "Đây là câu số 1.",
    "Đây là câu số 2.",
    "Câu số 3 khác với câu số 1.",
]
tokenizer.fit_on_texts(text_data)

input = "Đây là câu số 2"
# Mã hóa câu thành chuỗi số nguyên
encoded_data = tokenizer.texts_to_sequences(input)

# In ra các câu đã được mã hóa
print(encoded_data)