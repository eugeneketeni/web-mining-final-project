#Importing libraries
import pandas as pd
import nltk, string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

#full print display
pd.set_option('display.max_colwidth', -1)

data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/web-mining-final-project/master/Test_file.csv")
data.head()

data['text'] = data['text'].astype(str).str.lower()
text = data.loc[:, "text"]
# filter out punctuation 
table = str.maketrans({key: None for key in string.punctuation})
data['text'] = data['text'].apply(lambda x: x.translate(table))
#tokenize
data['text'] = [word_tokenize(text[i]) for i in range(len(text))]

#filter out stop words and custom words
stopwords = set(stopwords.words('english'))
customStopWords = set(["rt"])
data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stopwords])
data['text'] = data['text'].apply(lambda x: [item for item in x if item not in customStopWords])
#filter out links
data['text'] = data['text'].apply(lambda x: [item for item in x if not "http" in item])

print(data['text'])