
# coding: utf-8

# In[ ]:


#Importing libraries
import pandas as pd


# # Loading the dataset

# In[ ]:


data = pd.read_csv("https://raw.githubusercontent.com/eugeneketeni/web-mining-final-project/master/Test_file.csv")


# # Pre-processing the tweets

# ### Tokenization and removing Stopwords

# In[ ]:


import nltk
from nltk import word_tokenize, sent_tokenize


# In[1]:


stoplist = set(stopwords.words('english'))
tokenizer = data['text'].astype(str).apply(lambda line: [token for token in word_tokenize(line) if token not in stoplist])


# In[4]:


print(tokenizer)

