import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/web-mining-final-project/master/Test_file.csv")

import nltk
from nltk import word_tokenize, sent_tokenize

stoplist = set(stopwords.words('english'))
tokenizer = data['text'].astype(str).apply(lambda line: [token for token in word_tokenize(line) if token not in stoplist])

print(tokenizer)

