
# coding: utf-8

# In[41]:


import pandas as pd

data = pd.read_csv("Test_file.csv")

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()



data['text'] = data['text'].astype(str)

text = data.loc[:, "text"]
sentence = str(text)
data['text'] = [word_tokenize(text[i]) for i in range(len(text))]
print(data['text'])
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])










    


