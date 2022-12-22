#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import math


# In[2]:


# Read the csv file
# data = pd.read_csv('submissions.csv')


data = pd.read_csv('submissions.csv', encoding='utf-8', names = ['image_id','unixtime','rawtime','title','total_votes','reddit_id','number_of_upvotes',\
'subreddit','number_of_downvotes','localtime','score','number_of_comments','username',\
'undefined1','undefined2', 'undefined3'])
                
print(data)


# # Predictive Task 1:
# User Image interaction...
# using models:
#     Cosine Similarity
#     Jacard Similarity
#     pearson Similarity
#     
# 

# In[3]:


df = data[["image_id","username","total_votes"]]
user_image_tv=(df.to_numpy())[1:]
print(len(user_image_tv))
n=len(user_image_tv)

s=data[["title"]]


# In[4]:


trainset=user_image_tv[:3*n//4]
testset=user_image_tv[3*n//4:]


# In[5]:


usersPerImage = defaultdict(set) # Maps an item to the users who rated it
imagesPerUser = defaultdict(set) # Maps a user to the items that they rated
images = []
users = [] # To retrieve a rating for a specific user/item pair
votesPerInter={}

for d in trainset:
    u,i,v = d[0],d[1],d[2]
#     print(u,i,v)
    usersPerImage[i].add(u)
    imagesPerUser[u].add(i)
    votesPerInter[(u,i)] = v
    
# print(len(user_image_tv)) #132308
# print(len(votesPerInter)) #114779
#so for 132308-114779 # of (u,i) interaction is being repeated...
#for now considering the last assigned one
# print(len(usersPerImage)) #63338
# print(len(imagesPerUser)) #16737


# In[6]:


# imgCount = defaultdict(int)
# totalRead = 0

# for image,user,_ in user_image_tv:
#     imgCount[image] += 1
#     totalRead += 1

# mostPopular = [(bookCount[x], x) for x in bookCount]
# mostPopular.sort()
# mostPopular.reverse()


# In[7]:


def Jaccard(s1, s2):
    numerator = len(s1. intersection (s2))
    denominator = len(s1.union(s2))
    return numerator / denominator
    


# In[8]:


def CosineSet(s1, s2):
    # Not a proper implementation, operates on sets so correct for interactions only
    numer = len(s1.intersection(s2))
    denom = math.sqrt(len(s1)) * math.sqrt(len(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[9]:


def Pearson(i1, i2):
    # Between two items
    iBar1 = itemAverages[i1]
    iBar2 = itemAverages[i2]
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += (ratingDict[(u,i1)] - iBar1)*(ratingDict[(u,i2)] - iBar2)
    for u in inter: #usersPerItem[i1]:
        denom1 += (ratingDict[(u,i1)] - iBar1)**2
    #for u in usersPerItem[i2]:
        denom2 += (ratingDict[(u,i2)] - iBar2)**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[10]:


# def predictRating(user,item):
#     ratings = []
#     similarities = []
#     for d in imagesPerUser[user]:
#         if d == item: continue
#         ratings.append(d['star_rating'] - itemAverages[i2])
#         similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
#     if (sum(similarities) > 0):
#         weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
#         return itemAverages[item] + sum(weightedRatings) / sum(similarities)
#     else:
#         # User hasn't rated any similar items
#         return ratingMean


# In[12]:


def accuracy(predictions, y):
    correct=0
    for i in range(len(y)):
        if y[i]==predictions[i]:
            correct=correct+1
#     print(correct)
    acc = (correct) / len(y)
    #print(acc1)
    return acc


# # Cosine Similarity

# In[15]:


pred=[]
y=[]
for u,i,_ in testset:
    c=[0]
    y.append(1)
    for d in imagesPerUser[u]:
        if i==d: continue
        c.append(CosineSet(usersPerImage[d], usersPerImage[i]))
    k=max(c)
#     print(k)
    if k>0.5:
        pred.append(1)
    else:
        pred.append(0)

        
mse = mean_squared_error(y, pred)
print(len(y),len(pred))
print(mse)
print(accuracy(pred,y))
        


# # jaccard similarity

# In[16]:


pred=[]
y=[]
for u,i,_ in testset:
    c=[0]
    y.append(1)
    for d in imagesPerUser[u]:
        if i==d: continue
        c.append(Jaccard(usersPerImage[d], usersPerImage[i]))
    k=max(c)
#     print(k)
    if k>0.5:
        pred.append(1)
    else:
        pred.append(0)

msej = mean_squared_error(y, pred)
print(len(y),len(pred))
print(msej)
print(accuracy(pred,y))
        


# In[ ]:





# # BPR

# In[18]:


import gzip
import random
import scipy
import tensorflow as tf
from collections import defaultdict
from implicit import bpr
from surprise import SVD, Reader, Dataset
from sklearn.metrics import mean_squared_error
from surprise.model_selection import train_test_split
import tqdm
from tqdm import tqdm


# In[19]:


data = pd.read_csv('submissions.csv', encoding='utf-8', names = ['image_id','unixtime','rawtime','title','total_votes','reddit_id','number_of_upvotes',\
'subreddit','number_of_downvotes','localtime','score','number_of_comments','username',\
'undefined1','undefined2', 'undefined3'])


df = data[["username","image_id","total_votes"]]
f=df
df=df.dropna()
user_image_tv=(df.to_numpy())[1:]


# In[20]:


data = pd.read_csv('submissions.csv', encoding='utf-8', names = ['image_id','unixtime','rawtime','title','total_votes','reddit_id','number_of_upvotes',\
'subreddit','number_of_downvotes','localtime','score','number_of_comments','username',\
'undefined1','undefined2', 'undefined3'])
print(data)


# In[21]:


user_image_tv


# In[22]:


userIDs,itemIDs={},{}
for u,i,v in user_image_tv:
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)


# In[23]:


nUsers,nItem = len(userIDs),len(itemIDs)


# In[24]:


len(user_image_tv)


# In[25]:


print(len(f.dropna()))# != 
print(len(f))
f=f.dropna()


# In[26]:


nTrain = int(len(user_image_tv) * 0.9)
nTest = len(user_image_tv) - nTrain
interactionsTrain =  user_image_tv
interactionsTest =  user_image_tv[nTrain:]
# items = list(itemIDs.keys())
# interactionsTrain[0]
a=f.to_numpy()[1:]
d=[]
# print(f)
for i in a:
    a,b,c=i[0],i[1],i[2]
    d.append((a,b,c))
    


# In[27]:


# d


# In[28]:


itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)

for u,i,r in user_image_tv:
    itemsPerUser[u].append(i)
    usersPerItem[i].append(u)


# In[29]:


class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
#         if p<0.25:
#             return 0
#         else:
#             return 1
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))


# In[30]:


optimizer = tf.keras.optimizers.Adam(0.01)
modelBPR = BPRbatch(5, 0.000001)


# In[31]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[32]:


def trainingStepBPR(model, interactions):
    Nsamples = 1000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])
        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


# In[33]:


items = list(itemIDs.keys())


# In[34]:


for i in tqdm(range(1000)):
    obj = trainingStepBPR(modelBPR,d)


# In[35]:


userIDs['Mrpink12187']


# In[36]:


u,i,_ = interactionsTrain[0]
print(interactionsTrain[0])
# In this case just a score (that can be used for ranking), rather than a prediction of a rating
modelBPR.predict(userIDs[u], itemIDs[i]).numpy()


# In[37]:


2.4389625


# In[52]:


def accuracy(y_true,pred):  
    cnt=0
    for i in range(len(y_true)):
        if y_true[i]==pred[i]:
            cnt=cnt+1
    acc=cnt/len(pred)
#     print(acc)
    return acc


# In[53]:


# p=[]  
def check(u,i):
    temp=modelBPR.predict(userIDs[u],itemIDs[b]).numpy()
#     p.append(temp)
    if temp>0:
        return 1
    else:
        return 0
   


# In[54]:


interactionsTest
dtest = pd.DataFrame(interactionsTest, columns = ['user','image','votes'])


# In[55]:


len(interactionsTest)


# In[57]:


pred=[]
y_true=[]
for l in tqdm(interactionsTest):

    u,b = l[0],l[1]
    if u in userIDs and b in itemIDs:
         pred.append(check(u,b)) 
    y_true.append(1)


# In[58]:


print(accuracy(y_true,pred))


# In[ ]:





# In[38]:


# p=[]  
def check(u,i,thresh):
    temp=modelBPR.predict(userIDs[u],itemIDs[b]).numpy()
#     p.append(temp)
    if temp>thresh:
        return 1
    else:
        return 0
   


# In[39]:


interactionsTest
dtest = pd.DataFrame(interactionsTest, columns = ['user','image','votes'])
# print(interactionsTest)
# print(dtest)


# In[40]:


len(interactionsTest)


# In[47]:


def testing(t):
    pred=[]
    y_true=[]
    for l in (interactionsTest):

        u,b = l[0],l[1]
        if u in userIDs and b in itemIDs:
             pred.append(check(u,b,t)) 
        y_true.append(1)
    return pred,y_true
#     else:
#         if (b in return2):
#                 predictions.write(u + ',' + b + ",1\n")
#         else:
#                 predictions.write(u + ',' + b + ",0\n")
    
    
#         if (b in return2):
#              _ = predictions.write(u + ',' + b + ',' + str(1) + '\n')
#         ((u,b) in rrr) and (rrr[(u,b)] > 0.1) or
#         else:
#              _ = predictions.write(u + ',' + b + ',' + str(0) + '\n')
        
# predictions.close()


# In[48]:





# In[50]:


ay=[]
by=[]
dic={}
for i in tqdm(range(-4,17)):
    ay,by=testing(i)
    acc=accuracy(by,ay)
    dic[i]=acc
print(dic)


# In[ ]:




