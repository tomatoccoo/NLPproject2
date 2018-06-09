
# coding: utf-8

# In[1]:


import numpy as np
import pickle

import torch.utils.data as Data


# In[2]:

f = open('data.pkl', 'rb')
posts_ints = pickle.load(f)
labels = pickle.load(f)
vocab_to_int = pickle.load(f)
f.close()

# 将one-hot的编码转变为数字编码
labels = np.where(labels==1)[1]

# In[3]:

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

def get_batches(x, y, batch_size=64):
      n_batches = len(x) // batch_size
      x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
      for ii in range(0, len(x), batch_size):
            yield x[ii:ii + batch_size], y[ii:ii + batch_size]

class myData(Data.Dataset):
      def __init__(self, x, y):
            self.x = x
            self.y = y
      def __getitem__(self, index):
            return self.x[index], self.y[index]
      def __len__(self):
            return len(self.x)


BATCH_SIZE = 64
Train_set = myData(train_x, train_y)
Test_set = myData(test_x, test_y)
Train_loader = Data.DataLoader(dataset=Train_set, batch_size=BATCH_SIZE, shuffle=True)
Test_loader = Data.DataLoader(dataset=Test_set, batch_size=BATCH_SIZE, shuffle=True)



# In[7]:


# 定义RNN 结构
post_len = 50  #用户post条数
seq_len = 100  #每条post句子长度
hidden_size = 256 
lstm_layers = 1
batch_size_2 = 256  # 每次处理的用户批次
batch_size_1 = batch_size_2 * post_len
learning_rate = 0.01
n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1


# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN_CNN(nn.Module):
      def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super(RNN_CNN, self).__init__()

            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True).to(device)

            self.conv_1 = nn.Sequential( # in [batch, hidden_dim, 50]
                  nn.Conv1d(in_channels=self.hidden_dim, out_channels=32, kernel_size=9, stride=1, padding=(9-1)/2),
                  nn.Dropout(0.5),
                  nn.ReLU(),
                  nn.MaxPool1d(kernel_size=5)
            ).to(device)
            # [batch, 32, 10]

            self.conv_2 = nn.Sequential(
                  nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9,stride=1, padding=(9-1)/2),
                  nn.Dropout(0.5),
                  nn.ReLU(),
                  nn.MaxPool1d(kernel_size=2)
            ).to(device)
            # [batch, 32, 5]

            self.W = nn.Linear(32*5, 16).to(device)
            #[batch, 16]


      def forward(self, input):
            # input [batch, 50, each_post_length] [batch, 50, 100]

            input_trans = input.view(-1, 100) # [50*batch, 100]
            embeds = self.word_embeddings(input_trans).to(device)
            #embeds [ seq_len:100, 50*batch, embedding_lens]

            h_t = torch.zeros(1, input.size(0)*50, self.hidden_dim).to(device)# [1, batch_size, hidden_size]
            c_t = torch.zeros(1, input.size(0)*50, self.hidden_dim).to(device)

            out, (h_t, c_t) = self.lstm(embeds, (h_t.detach(), c_t.detach())) #[1, batch_size*50, hidden_size
            rnn_out = h_t.view(input.size(0), 50, self.hidden_dim) # [batch,  channel, 50*hidden_dim:256*50]

            rnn_out = rnn_out.permute(0, 2, 1)
            conv1_out = self.conv_1(rnn_out)
            conv2_out = self.conv_2(conv1_out)

            out = conv2_out.view(input.size(0), -1)
            out = self.W(out)
            out = F.log_softmax(out, dim=1)
            return out


class RNN_RNN(nn.Module):
      def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super(RNN_RNN, self).__init__()

            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True).to(device)
            self.lstm2 = nn.LSTM(hidden_dim, hidden_dim,batch_first=True).to(device)
            self.W = nn.Linear(hidden_dim, 16).to(device)
            # [batch, 16]

      def forward(self, input):
            # input [batch, 50, each_post_length] [batch, 50, 100]
            input_trans = input.view(-1,100)#[50*batch, 100]
            embeds = self.word_embeddings(input_trans).to(device)
           #[50*batch,100， embedding_dim]
            h_t_0 = torch.zeros(1, input.size(0) * 50, self.hidden_dim).to(device)  # [1, batch_size*50, hidden_size]
            c_t_0 = torch.zeros(1, input.size(0) * 50, self.hidden_dim).to(device)

            out, (h_t, c_t) = self.lstm(embeds, (h_t_0.detach(), c_t_0.detach()))
            h_t =  h_t.view(input.size(0), 50, self.hidden_dim)

            h_t_1 = torch.zeros(1, input.size(0), self.hidden_dim).to(device)  # [1, batch_size, hidden_size]
            c_t_1 = torch.zeros(1, input.size(0), self.hidden_dim).to(device)

            out, (ht, c_t) = self.lstm2(h_t, (h_t_1.detach(), c_t_1.detach()))
            out = self.W(ht.view(input.size(0), self.hidden_dim))
            out = F.log_softmax(out, dim=1)
            return out

class RNN_Linear(nn.Module):
      def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super(RNN_Linear, self).__init__()
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim).to(device)
            self.W = nn.Linear(hidden_dim*50, 16).to(device)
            # [batch, 16]

      def forward(self, input):
            # input [batch, 50, each_post_length] [batch, 50, 100]

            input_trans = input.view(-1, 100)  # [50*batch, 100]
            embeds = self.word_embeddings(input_trans).to(device)
            # [100, 50*batch, embedding_dim]
            h_t_0 = torch.zeros(1, input.size(0) * 50, self.hidden_dim).to(device)  # [1, batch_size*50, hidden_size]
            c_t_0 = torch.zeros(1, input.size(0) * 50, self.hidden_dim).to(device)
            out, (h_t, c_t) = self.lstm(embeds, (h_t_0.detach(), c_t_0.detach()))
            h_t = h_t.view(input.size(0), self.hidden_dim*50)

            out = self.W(h_t)
            out = F.log_softmax(out, dim=1)
            return out


model = RNN_CNN(embedding_dim = 128,
                hidden_dim = 256,
                vocab_size = 142530,
                tagset_size = 16)
loss_function = nn.NLLLoss()

def train(EPOCH = 10):
      test()
      model.train()
      for e in range(EPOCH):
            optimizer = optim.Adam(model.parameters(), lr=max(0.01*0.5**e, 1e-6))#减小学习率
            for ii, (x, y) in enumerate(Train_loader):
                  model.zero_grad()

                  input = x.long()
                  target = y.to(device)

                  output = model(input)
                  loss = loss_function(output, target)
                  loss.backward()
                  optimizer.step()

                  if ii % 10 == 0:
                        pred = output.max(1)[1]
                        correct = pred.eq(target).sum().cpu().numpy()

                        print("Epoch:{}".format(e),
                              "step:{}".format(ii),
                              "correct:{}/{}".format(correct, len(target)),
                              "loss:{:.4f}".format(loss.cpu().item()))
            test()
            model.train()
      print("train complete")

def test():
      with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for ii, (x, y) in enumerate(Test_loader):
                  input = x.long()
                  target = y.to(device)
                  output = model(input)
                  pred = output.max(1)[1]
                  correct = correct + pred.eq(target).sum().cpu().numpy()
                  total = total + len(input)

            print('test: {}/{}, rate:{:.4f}'.format(correct, total, correct/total))

train(EPOCH=50)
torch.save(model.state_dict(), 'RNN_CNN_model.params')