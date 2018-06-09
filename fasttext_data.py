import numpy as np
import pandas as pd
import pickle

text=pd.read_csv("../input/mbti_1.csv" ,index_col='type') #type列为索引列

labels=text.index.tolist()

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
    # print('把每个用户的50条推送拆开: ',len(things))

    # 分别处理每一条推送
    # Remove URLs, links etc，将每条链接post转换成URL
    for t in things:
        t = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            'URL', t, flags=re.MULTILINE)
        # 移除标点
        puncs = ['.', '?', '!', '\n', '@', '#', '$', '^', '%', '*', '&', '(', ')', '-', '_', '+', '=', '{', '}', '[',
                 ']', '|', '\\', '"', "'", ';', ':', '<', '>', '/']
        for punc in puncs:
            t = t.replace(punc, '')
            # Remove extra white spaces
        t = re.sub('\s+', ' ', t).strip()
        # This would have removed most of the links but probably not all
        res.append(t)
    return res

posts=text.posts.tolist()
posts=[post_cleaner(post) for post in posts]

f = open('fasttext_data.txt', 'w')
for post in posts:
    f.write(' '.join(post))
f.close()


labels=text.index.tolist()
f1 = open('fasttext_train_data.txt', 'w')
f2 = open('fasttext_test_data.txt', 'w')

for ii, post in enumerate(posts):
    if ii<len(posts)*0.8:
        f1.writelines('__label__'+labels[ii]+ ' ' + ' '.join(post)+'\n')
    else:
        f2.writelines('__label__'+labels[ii]+ ' ' + ' '.join(post)+'\n')
f1.close()
f2.close()



