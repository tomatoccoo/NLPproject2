# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


# 将50个句子拼成一句话，并存为data2
import numpy as np
import pandas as pd
import pickle
import torch


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# load dataset
text=pd.read_csv("../input/mbti_1.csv" ,index_col='type')
print(text.shape)
print(text[0:5])
print(text.iloc[2])

# One hot encode labels
labels = text.index.tolist()
mbti_to_int={'INFP':0, 'INFJ':1, 'INTP':2, 'INTJ':3, 'ENTP':4,'ENFP':5,'ISTP':6,'ISFP':7,'ENTJ':8,'ISTJ':9,'ENFJ':10,'ISFJ':11,'ESTP':12,'ESFP':13,'ESFJ':14,'ESTJ':15}
labels = torch.tensor([mbti_to_int[type] for type in labels])
labels = torch.eye(16).index_select(dim=0, index=labels)
labels = labels.numpy()


import re


# Function to clean data ... will be useful later
def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    # Covert all uppercase characters to lower case
    post = post.lower()

    # Remove |||
    post = post.replace('|||', "")

    # Remove URLs, links etc
    post = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        '', post, flags=re.MULTILINE)
    # This would have removed most of the links but probably not all

    # Remove puntuations
    puncs1 = ['@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', '"', "'",
              ';', ':', '<', '>', '/']
    for punc in puncs1:
        post = post.replace(punc, '')

    puncs2 = [',', '.', '?', '!', '\n']
    for punc in puncs2:
        post = post.replace(punc, ' ')
        # Remove extra white spaces
    post = re.sub('\s+', ' ', post).strip()
    return post

# Clean up posts
# Covert pandas dataframe object to list. I prefer using lists for prepocessing.
posts=text.posts.tolist()
posts=[post_cleaner(post) for post in posts]


# Count total words
from collections import Counter

word_count=Counter()
for post in posts:
    word_count.update(post.split(" "))

# Size of the vocabulary available to the RNN
vocab_len=len(word_count)
print(vocab_len)

print(len(posts[0]))


vocab = sorted(word_count, key=word_count.get, reverse=True)
# Create your dictionary that maps vocab words to integers here
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

posts_ints=[]
for post in posts:
    posts_ints.append([min(vocab_to_int[word], 8000-1) for word in post.split()])

print(posts_ints[0])
print(len(posts_ints[0]))

seq_len = 500
features = np.zeros((len(posts_ints), seq_len), dtype=int)
for i, row in enumerate(posts_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
print(features[:10])



f = open('data2.pkl', 'wb')
pickle.dump(features, f)
pickle.dump(labels, f)
pickle.dump(vocab_to_int, f)
f.close()

