
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


# In[ ]:


# load dataset
text=pd.read_csv("mbti_1.csv" ,index_col='type') #type列为索引列


# In[ ]:


#preprocessing labels
from sklearn.preprocessing import LabelBinarizer

# One hot encode labels
labels=text.index.tolist()
encoder=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
labels=encoder.fit_transform(labels)#以16个不同的label为维度，one hot encode
print(labels.shape)
labels=np.array(labels)


# In[ ]:


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


# In[ ]:


# Clean up posts
# Covert pandas dataframe object to list. I prefer using lists for prepocessing.
# 预处理
posts=text.posts.tolist()
posts=[post_cleaner(post) for post in posts]
print(len(posts))


# In[ ]:


# Count total words
from collections import Counter

word_count=Counter()
for post in posts:
    for thing in post:
        word_count.update(thing.split(" "))
    
# Size of the vocabulary available to the RNN    
vocab_len=len(word_count)


# In[ ]:


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
        thing_ints.append([vocab_to_int[word] for word in thing.split()])
    posts_ints.append(thing_ints)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#取单条post长度为100来padding
seq_len = 100  

for post in posts_ints:
    for thing in post:
        #print(thing)
        while(len(thing) < seq_len):
            thing.insert(0,0)
        while(len(thing) > seq_len):
            thing.pop(-1)


# In[ ]:


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


# In[ ]:


# 定义RNN 结构
post_len = 50  #用户post条数
seq_len = 100  #每条post句子长度
hidden_size = 256 
lstm_layers = 1
batch_size_2 = 256  # 每次处理的用户批次
batch_size_1 = batch_size_2 * post_len
learning_rate = 0.01
embed_dim=100 
n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1



# In[ ]:



# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    input_data = tf.placeholder(tf.int32, [None, None,None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# In[ ]:


# 随机初始化 Word Embedding，然后在训练过程中不断更新embedding矩阵的值
with graph.as_default():
    # 随机初始化ids-embedding查找表
    # 定义第一层RNN
  
    with tf.variable_scope("rnn_1"):   
       
        embedding= tf.Variable(tf.random_uniform(shape=(n_words,embed_dim),minval=-1,maxval=1))
        embed_tmp=tf.nn.embedding_lookup(embedding,input_data) 
  
        #(256,50,100,100)to(256*50,100,100)
        embed = tf.reshape(embed_tmp,[batch_size_1,seq_len,embed_dim])

        # basic LSTM cell
        lstm_1 = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    
        # Add dropout to the cell
        drop_1 = tf.contrib.rnn.DropoutWrapper(lstm_1,output_keep_prob=keep_prob)
    
        # Stack up multiple LSTM layers, for deep learning
        cell_1 = tf.contrib.rnn.MultiRNNCell([drop_1]* lstm_layers)
    
        # Getting an initial state of all zeros
        initial_state_1 = cell_1.zero_state(batch_size_1, tf.float32)
    
        # Output和每一层的state,取最后一层hidden unit作为句子的向量表示(256*50,256)
        outputs_1,final_state_1=tf.nn.dynamic_rnn(cell_1,embed,initial_state=initial_state_1,dtype=tf.float32 )


# In[ ]:


# 定义第二层RNN
with graph.as_default():
    
    with tf.variable_scope("rnn_2"):
        # basic LSTM cell
        lstm_2 = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    
        # Add dropout to the cell
        drop_2 = tf.contrib.rnn.DropoutWrapper(lstm_2,output_keep_prob=keep_prob)
    
        # Stack up multiple LSTM layers, for deep learning
        cell_2 = tf.contrib.rnn.MultiRNNCell([drop_2]* lstm_layers)
    
        # Getting an initial state of all zeros
        initial_state_2 = cell_2.zero_state(batch_size_2, tf.float32)
         
        # Output和每一层的state,以sentence-embedding构成的矩阵作为输入shape:(256,50,256)
        #(256*50,256)to(256,50,256)
        sentence_embedding = tf.reshape(final_state_1[-1][-1],[batch_size_2,post_len,hidden_size])
    
        outputs_2,final_state_2=tf.nn.dynamic_rnn(cell_2,sentence_embedding,initial_state=initial_state_2,dtype=tf.float32)   


# In[ ]:


# 定义loss
with graph.as_default():
    # 全连接和softmax
    pre = tf.layers.dense(outputs_2[:,-1], 16, activation=tf.nn.relu)
    predictions=tf.layers.dense(pre, 16, activation=tf.nn.softmax)
    
    cost = tf.losses.mean_squared_error(labels_, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
  
    
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[ ]:


def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# In[ ]:


# Training
epochs = 2

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        #state = sess.run(initial_state_2)
        #训练集
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size_2), 1):

            feed = {input_data: x,
                    labels_: y,
                    keep_prob: 0.5,
                    }
            loss, state, _ = sess.run([cost, final_state_2, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))
            iteration +=1
        #验证集   
        val_acc = []
        #val_state = sess.run(cell_2.zero_state(batch_size_2, tf.float32))
        for x, y in get_batches(val_x, val_y, batch_size_2):
            feed = {input_data: x,
                    labels_: y,
                    keep_prob: 0.5,
                    }
            batch_acc, val_state = sess.run([accuracy, final_state_2], feed_dict=feed)
            val_acc.append(batch_acc)
        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                               
    saver.save(sess, "checkpoints4/mbti.ckpt")
print('----END----')


# In[ ]:


# Testing

test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints4'))
    #test_state = sess.run(cell_2.zero_state(batch_size_2, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size_2), 1):
        feed = {input_data: x,
                labels_: y,
                keep_prob: 1,
                }
        batch_acc, test_state = sess.run([accuracy, final_state_2], feed_dict=feed)
        test_acc.append(batch_acc)    
    print("  全数据集 Test accuracy: {:.3f}".format(np.mean(test_acc)))


