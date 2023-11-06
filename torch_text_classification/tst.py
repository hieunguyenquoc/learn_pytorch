import pandas as pd
import re 
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

class Preprocessing:
    def tokenization(self,X_train):
        self.tokens = Tokenizer(num_words=1000)
        self.tokens.fit_on_texts(X_train)

    def sequence_to_token(self,input):
        sequences = self.tokens.texts_to_sequences(input)
        return pad_sequences(sequences, maxlen = 20)

X_train = ['70 years ago the first atomic attack flattened #Hiroshima 3 days later it was #Nagasaki both war crimes to put Moscow in its place'     
 'Contruction upgrading ferries to earthquake standards in Vashon Mukilteo: The upgrades will bring the vulnera... http://t.co/Au5jWGT0ar'
 'Just realized that maybe it not normal to sit up front with an Uber driver? Panicking']

X_train_arr = np.array(X_train)
preprocess = Preprocessing()
preprocess.tokenization()
preprocess.sequence_to_token()




