import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk, string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#Test data
#data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/web-mining-final-project/master/Test_file.csv")

#Full data
data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/webmining/master/tweets.csv")

#Turn off output print limit
#pd.set_option('display.max_colwidth', -1)

print(data['text'].head())
data['text'] = data['text'].astype(str).str.lower()
text = data.loc[:, "text"]
table = str.maketrans({key: None for key in string.punctuation})
stopwords = set(stopwords.words('english'))
customStopWords = set(["rt",","])
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.ADV)
    
def processRow(row):
    processedRow = []

    #Drop punctuation
    row = row.translate(table)

    #tokenize 
    tokens = nltk.word_tokenize(row)

    #Filter out stopwords, customwords, and links
    for word in tokens:
        if (word not in stopwords and word not in customStopWords and
            'http' not in word):
            word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            processedRow.append(word)

    return processedRow

data['text'] = data['text'].apply(processRow)
print(data['text'].head())
