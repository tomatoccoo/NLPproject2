
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[2]:


# load dataset
text=pd.read_csv("../input/mbti_1.csv" ,index_col='type') #type列为索引列


# In[3]:


#preprocessing labels
from sklearn.preprocessing import LabelBinarizer

# One hot encode labels
labels=text.index.tolist()
encoder=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
labels=encoder.fit_transform(labels)#以16个不同的label为维度，one hot encode
print(labels.shape)
labels=np.array(labels)


# In[4]:


#preprocessing posts
import re

# Function to clean data..
def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    res = []
    # Covert all uppercase characters to lower case
    post = post.lower()   
    
    things = post.split(r'|||')      
    #print('把每个用户的50条推送拆开: ',len(things))

    # 分别处理每一条推送
    # Remove URLs, links etc，将每条链接post转换成URL
    for t in things:
        t = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', 'URL', t, flags=re.MULTILINE) 
        # 移除标点
        puncs=['.','?','!','\n','@','#','$','^','%','*','&','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
        for punc in puncs:
            t=t.replace(punc,'') 
        # Remove extra white spaces
        t=re.sub( '\s+', ' ', t ).strip()
        # This would have removed most of the links but probably not all 
        res.append(t)
    return res


# In[5]:


# Clean up posts
# Covert pandas dataframe object to list. I prefer using lists for prepocessing.
# 预处理
posts=text.posts.tolist()
posts=[post_cleaner(post) for post in posts]
print(len(posts))


# In[6]:


# Count total words
from collections import Counter

word_count=Counter()
for post in posts:
    for thing in post:
        word_count.update(thing.split(" "))
    
# Size of the vocabulary available to the RNN    
vocab_len=len(word_count)


# In[7]:


# Convert words to integers

# Create a look up table
vocab = sorted(word_count, key=word_count.get, reverse=True)

# Create your dictionary that maps vocab words to integers here
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}# 序号从1开始，保留0

# 把posts中的字符转换为词表中的序号
posts_ints=[]
for post in posts:
    thing_ints =[]
    for thing in post:
        thing_ints.append([min(vocab_to_int[word],142530-1) for word in thing.split()])
    posts_ints.append(thing_ints)


# In[8]:


# Make posts uniform

# 计算用户推送的平均长度
posts_lens = Counter([len(x) for x in posts_ints]) #取第3层而不是第2层 
print('用户推送的平均次数 :  ',np.mean(list(posts_lens.keys()))) #50

"""
每个{}代表一个用户，都包含50个左右的post,每个最里层[]代表的是一条推送，长度不一样
posts[  {[],[],[]},
        {        },
        {        },
     ...
     ]
"""
#计算用户单条post的平均长度
thing_lens=Counter()
for post in posts:
    thing_lens.update([len(x) for x in post])
    
# Size of the vocabulary available to the RNN    
print("用户单条推送的平均 : ", np.mean(list(thing_lens.keys()))) 



# In[9]:


# 按照用户推送句子的长度中间值来padding
post_len = 50  #取post个数为50
padding = []
#print(len(posts_ints))

tmp_posts = posts_ints

for row in tmp_posts:

    while(len(row) < post_len):
        row.insert(0,padding)
    while(len(row) > post_len):
        row.pop(-1)
    #print(len(row))


# In[10]:


#取单条post长度为100来padding
seq_len = 100  

for post in posts_ints:
    for thing in post:
        #print(thing)
        while(len(thing) < seq_len):
            thing.insert(0,0)
        while(len(thing) > seq_len):
            thing.pop(-1)


# In[11]:


f = open('data.pkl', 'wb')
pickle.dump(posts_ints, f)
pickle.dump(labels, f)
pickle.dump(vocab_to_int,f)
f.close()


# In[12]:


#preparing training,test and validation datasets

split_frac = 0.8
features = np.array(posts_ints)
num_ele=int(split_frac*len(features))
rem_ele=len(features)-num_ele
train_x, val_x = features[:num_ele],features[num_ele: num_ele + int(rem_ele/2)]
train_y, val_y = labels[:num_ele],labels[num_ele:num_ele + int(rem_ele/2)]

test_x =features[num_ele + int(rem_ele/2):]
test_y = labels[num_ele + int(rem_ele/2):]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
#print(len(train_x)+len(val_x)+len(test_x))


# In[29]:




