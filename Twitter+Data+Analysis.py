import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk, string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import csv
#Test data
#data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/web-mining-final-project/master/Test_file.csv")

#Full data
data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/webmining/master/tweets.csv")

#Turn off output print limit
pd.set_option('display.max_colwidth', -1)


print(data['text'].head())
data['text'] = data['text'].astype(str)

#Working and Scoring
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    return score
# iterating over rows for specific words to get sensitivity scores (positive or negative) and writing to a csv
for index, row in data.iterrows():
    if 'clinton' in row['text']:
        print(sentiment_analyzer_scores(row['text']))
        sentiment = sentiment_analyzer_scores(row['text'])
        pos = sentiment['pos']
        neg = sentiment['neg']
        with open('clinton_sentiment.csv','a') as fd:
            fields = [pos, neg]
            writer = csv.writer(fd)
            writer.writerow(fields)
    if 'trump' in row['text']:
        print(sentiment_analyzer_scores(row['text']))
        sentiment = sentiment_analyzer_scores(row['text'])
        pos = sentiment['pos']
        neg = sentiment['neg']
        with open('trump_sentiment.csv','a') as fd:
            fields = [pos, neg]
            writer = csv.writer(fd)
            writer.writerow(fields)
    if 'russia' in row['text']:
        print(sentiment_analyzer_scores(row['text']))
        sentiment = sentiment_analyzer_scores(row['text'])
        pos = sentiment['pos']
        neg = sentiment['neg']
        with open('russia_sentiment.csv','a') as fd:
            fields = [pos, neg]
            writer = csv.writer(fd)
            writer.writerow(fields)


table = str.maketrans({key: None for key in string.punctuation})
stopwords = set(stopwords.words('english'))
customStopWords = set(["rt",","])
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    #Map POS tag to first character lemmatize() accepts
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








