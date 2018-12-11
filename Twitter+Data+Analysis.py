
import pandas as pd

data = pd.read_csv("C:\\Users\\Eugene\\Downloads\\Final_Project\\web-mining-final-project\\Test_file.csv")

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk, string

pd.set_option('display.max_colwidth', -1)
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

print(data['text'].head())

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.ADJ)
lemmatizer = WordNetLemmatizer()
sentence = str(text)
print(data['text'])
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
