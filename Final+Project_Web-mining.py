# coding: utf-8
# In[ ]:
#Importing libraries
import pandas as pd
# # Loading the dataset
# In[ ]:
data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/web-mining-final-project/master/Test_file.csv")
# In[ ]:
#data.head()
# # Pre-processing the tweets
# ### Tokenization
# In[ ]:
import nltk
from nltk import word_tokenize, sent_tokenize
# In[ ]:
data = pd.DataFrame(data['text'])
# In[ ]:
data['text'] = data['text'].astype(str)
# In[ ]:
text = data.loc[:, "text"]
tokenizer = [word_tokenize(text[i]) for i in range(len(text))]
print(tokenizer)
# In[ ]:
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
print(stopwords)
# In[ ]:
words = [word_tokenize(text[i]) for i in range(len(text))]
filtered_sentence = []
for w in words:
    if w not in stopwords:
        filtered_sentence.append(w)
print(filtered_sentence)
