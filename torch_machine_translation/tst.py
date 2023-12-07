import string
def remove_punctuation(text):
    # Loại bỏ các ký tự dấu câu
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    return text

# Ví dụ sử dụng
text_with_punctuation = "Chào bạn, đây là một câu với dấu câu! % ^ & *"
text_without_punctuation = remove_punctuation(text_with_punctuation)

print("Original text:", text_with_punctuation)
print("Text without punctuation:", text_without_punctuation)