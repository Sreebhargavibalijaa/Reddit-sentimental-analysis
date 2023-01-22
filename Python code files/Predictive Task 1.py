#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import pandas as pd
import numpy as np


# In[3]:


#cleanedList = [x for x in countries if str(x) != 'nan']


# In[4]:


data = pd.read_csv('submissions.csv', encoding='utf-8', names = ['image_id','unixtime','rawtime','title','total_votes','reddit_id','number_of_upvotes',\
'subreddit','number_of_downvotes','localtime','score','number_of_comments','username',\
'undefined1','undefined2', 'undefined3'])
print(len(data))
data=data[['total_votes','number_of_upvotes','number_of_downvotes','score','number_of_comments']]
data=data.dropna()
print(len(data))

rr=data[['total_votes','number_of_upvotes','number_of_downvotes','score','number_of_comments']]
print(len(rr))
rr=rr.dropna()
print(len(rr))
rr.corr()
# print(data['total_votes'].corr(data['number_of_comments']))
# print(data['score'].corr(data['number_of_comments']))


# In[5]:


r=data[['total_votes']].to_numpy()
tv=[]
print(len(data))
for i in r[1:]:
    tv.append(int(i[0]))
print(len(tv))         #dropped 2 nan values


# In[6]:


r2=data[["number_of_comments"]].to_numpy()
c=[]
print(len(data))
for i in r2[1:]:
    c.append(int(i[0]))
print(len(c))         #dropped 2 nan values


# In[7]:


r3=data[['score']].to_numpy()
s=[]
print(len(data))
for i in r3[1:]:
    s.append(int(i[0]))
print(len(s))         #dropped 2 nan values


# In[8]:


r4=data[['number_of_upvotes']].to_numpy()
u=[]
print(len(data))
for i in r4[1:]:
    u.append(int(i[0]))


# In[9]:


r5=data[['number_of_downvotes']].to_numpy()
d=[]
print(len(d))
for i in r5[1:]:
    d.append(int(i[0]))
print(len(d))


# In[10]:


matrix_tv_s = np.corrcoef(tv, s)
print(matrix_tv_s )


# In[11]:


matrix_u_c = np.corrcoef(u, c)
print(matrix_u_c )


# In[12]:


matrix_u_s = np.corrcoef(u, s)
print(matrix_u_s )


# In[13]:


matrix_tv_c = np.corrcoef(tv, c)
print(matrix_tv_c)


# In[14]:


matrix_s_c = np.corrcoef(s, c)
print(matrix_s_c)


# # Correlation Matrix

# In[15]:


# months = ['Jan','Apr','Mar','June']
# days = [31, 30, 31, 30]

# Make dictionary, keys will become dataframe column names
intermediate_dictionary = {'total_votes':tv,'number_of_upvotes':u,'score':s,'number_of_comments':c}

# Convert dictionary to Pandas dataframe
impcols= pd.DataFrame(intermediate_dictionary)

print(impcols)


# In[16]:


impcols.corr()


# In[17]:


impcols.corr()

import seaborn as sns
import pandas as pd
import numpy as np

# Create a dataset
# impcols
# Default heatmap
p1 = sns.heatmap(impcols.corr(),annot = True, fmt= '.4f')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


# In[21]:


model_performance_dict = dict()


# In[22]:


# print(type(data[['total_votes']]))
#     print(int(i[0]))
#  print((data[['total_votes']][1:].to_numpy().tolist()))
# r=pd.to_numeric(data[['total_votes']][1:].to_numpy().tolist())


# In[23]:


X_train=u[:len(u)*3//4]
y_train=s[:len(s)*3//4]

X_test=u[len(u)*3//4:]
y_test=s[len(s)*3//4:]


# # upvotes and scores

# In[24]:


def model_diagnostics(model, pr=True):
    """
    Returns and prints the R-squared, RMSE and the MAE for a trained model
    """
    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    if pr:
        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")
    
    return [r2,np.sqrt(mse),mae]


# # Baseline Model

# In[25]:


baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train,y_train)
model_performance_dict["Baseline"] = model_diagnostics(baseline)


# # Linear Regressor
# 

# In[26]:


X = np.matrix ([[1 ,l] for l in X_train ])
y = np.matrix(y_train).T
# print(X)
# print(y)
Xtst=np.matrix ([[1 ,l] for l in X_test ])
ytst = np.matrix(y_test).T
# print(y.shape)
# print(X.shape)

import sklearn
model = sklearn. linear_model . LinearRegression ( fit_intercept =False)
model.fit(X,y)

pred=model.predict(Xtst)
mse = mean_squared_error(ytst,pred)
rmse=np.sqrt(mse)
r2 = r2_score(ytst, pred)
print((rmse))
mae = mean_absolute_error(ytst, pred)
model_performance_dict["Linear Regression"] = [r2,rmse,mae]
# linear = LinearRegression()

# linear.fit(X,y)
# model_performance_dict["Linear Regression"] = model_diagnostics(linear)


# # Lasso Regression

# In[27]:


lasso = LassoCV(cv=30).fit(X, y)

predl=lasso.predict(Xtst)
msel = mean_squared_error(ytst,predl)
rmsel=np.sqrt(msel)
r2l = r2_score(ytst, predl)
print((rmsel))
mael = mean_absolute_error(ytst, predl)
model_performance_dict["Lasso Regression"] = [r2l,rmsel,mael]


# model_performance_dict["Lasso Regression"] = model_diagnostics(lasso)


# # Ridge Regression

# In[28]:


ridge = RidgeCV(cv=10).fit(X,y)

pred_r=ridge.predict(Xtst)
mse_r = mean_squared_error(ytst,pred_r)
rmse_r=np.sqrt(mse_r)
r2_r = r2_score(ytst, pred_r)
print((rmse_r))
mae_r = mean_absolute_error(ytst, pred_r)
model_performance_dict["Ridge Regression"] = [r2_r,rmse_r,mae_r]

# model_performance_dict["Ridge Regression"] = model_diagnostics(ridge)


# # Elastic Net Regression

# In[29]:


elastic_net = ElasticNetCV(cv = 30).fit(X,y)

pred_e=elastic_net.predict(Xtst)
mse_e = mean_squared_error(ytst,pred_e)
rmse_e=np.sqrt(mse_e)
r2_e= r2_score(ytst, pred_e)
print((rmse_e))
mae_e= mean_absolute_error(ytst, pred_e)
model_performance_dict["Elastic Net Regression"] = [r2_e,rmse_e,mae_e]
# model_performance_dict["Elastic Net Regression"] = model_diagnostics(elastic_net)


# # K-Nearest Neighbor Regression
# 
# 

# In[30]:


knr = KNeighborsRegressor()
knr.fit(X, y)

pred_k=knr.predict(Xtst)
mse_k= mean_squared_error(ytst,pred_k)
rmse_k=np.sqrt(mse_k)
r2_k= r2_score(ytst, pred_k)
print((rmse_k))
mae_k= mean_absolute_error(ytst, pred_k)
model_performance_dict["KNN Regression"] = [r2_k,rmse_k,mae_k]
# model_performance_dict["KNN Regression"] = model_diagnostics(knr)


# # DecisionTreeRegressor

# In[31]:


dt = DecisionTreeRegressor(min_samples_split=45, min_samples_leaf=45, random_state = 10)
dt.fit(X,y)


pred_dtr=dt.predict(Xtst)
mse_dtr= mean_squared_error(ytst,pred_dtr)
rmse_dtr=np.sqrt(mse_dtr)
r2_dtr= r2_score(ytst, pred_dtr)
print((rmse_dtr))
mae_dtr= mean_absolute_error(ytst, pred_dtr)
model_performance_dict["Decision Tree"] = [r2_dtr,rmse_dtr,mae_dtr]

# model_performance_dict["Decision Tree"] = model_diagnostics(dt)


# # RandomForestRegressor

# In[32]:


rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
rf.fit(X, y)


pred_rfr=rf.predict(Xtst)
mse_rfr= mean_squared_error(ytst,pred_rfr)
rmse_rfr=np.sqrt(mse_rfr)
r2_rfr= r2_score(ytst, pred_rfr)
print((rmse_rfr))
mae_rfr= mean_absolute_error(ytst, pred_rfr)
model_performance_dict["Random Forest"] = [r2_rfr,rmse_rfr,mae_rfr]


# model_performance_dict["Random Forest"] = model_diagnostics(rf)


# # Gradient Boosting Regression

# In[33]:


gbr = GradientBoostingRegressor(n_estimators=70, max_depth=5)
gbr.fit(X, y)


pred_gbr=gbr.predict(Xtst)
mse_gbr= mean_squared_error(ytst,pred_gbr)
rmse_gbr=np.sqrt(mse_gbr)
r2_gbr= r2_score(ytst, pred_gbr)
print((rmse_gbr))
mae_gbr= mean_absolute_error(ytst, pred_gbr)
model_performance_dict["Gradient Boosting"] = [r2_gbr,rmse_gbr,mae_gbr]


# model_performance_dict["Gradient Boosting Regression"] = model_diagnostics(gbr)


# In[34]:


for i in model_performance_dict:
    print(i,model_performance_dict[i])


# In[35]:


d=[] 
for i in model_performance_dict:
        k=model_performance_dict[i]
        b=[i]
        a=b+k
        print(a)
        d.append(a)
# print(d)
table1= pd.DataFrame(d,columns = ["model","r2","rmse","mae"])
table1


# In[ ]:





# In[ ]:





# In[ ]:





# # Total Votes and Comments

# In[36]:


X_train=tv[:len(u)*3//4]
y_train=c[:len(s)*3//4]

X_test=tv[len(u)*3//4:]
y_test=c[len(s)*3//4:]


X = np.matrix ([[1 ,l] for l in X_train])
y = np.matrix(y_train).T
# print(X)
# print(y)
Xtst=np.matrix ([[1 ,l] for l in X_test])
ytst = np.matrix(y_test).T
# print(y.shape)
# print(X.shape)
model_performance=dict()


# In[37]:


def model_diagnostics(model, pr=True):
    """
    Returns and prints the R-squared, RMSE and the MAE for a trained model
    """
    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    if pr:
        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")
    return [r2,np.sqrt(mse),mae]

# Baseline Model
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train,y_train)
model_performance["Baseline"] = model_diagnostics(baseline)

# Linear Regressor
import sklearn
model = sklearn. linear_model . LinearRegression ( fit_intercept =False)
model.fit(X,y)
pred=model.predict(Xtst)
mse = mean_squared_error(ytst,pred)
rmse=np.sqrt(mse)
r2 = r2_score(ytst, pred)
print((rmse))
mae = mean_absolute_error(ytst, pred)
model_performance["Linear Regression"] = [r2,rmse,mae]

#Lasso Regression
lasso = LassoCV(cv=30).fit(X, y)
predl=lasso.predict(Xtst)
msel = mean_squared_error(ytst,predl)
rmsel=np.sqrt(msel)
r2l = r2_score(ytst, predl)
print((rmsel))
mael = mean_absolute_error(ytst, predl)
model_performance["Lasso Regression"] = [r2l,rmsel,mael]


# Ridge Regression
ridge = RidgeCV(cv=10).fit(X,y)
pred_r=ridge.predict(Xtst)
mse_r = mean_squared_error(ytst,pred_r)
rmse_r=np.sqrt(mse_r)
r2_r = r2_score(ytst, pred_r)
print((rmse_r))
mae_r = mean_absolute_error(ytst, pred_r)
model_performance["Ridge Regression"] = [r2_r,rmse_r,mae_r]


# Elastic Net Regression
elastic_net = ElasticNetCV(cv = 30).fit(X,y)
pred_e=elastic_net.predict(Xtst)
mse_e = mean_squared_error(ytst,pred_e)
rmse_e=np.sqrt(mse_e)
r2_e= r2_score(ytst, pred_e)
print((rmse_e))
mae_e= mean_absolute_error(ytst, pred_e)
model_performance["Elastic Net Regression"] = [r2_e,rmse_e,mae_e]


# K-Nearest Neighbor Regression
knr = KNeighborsRegressor()
knr.fit(X, y)
pred_k=knr.predict(Xtst)
mse_k= mean_squared_error(ytst,pred_k)
rmse_k=np.sqrt(mse_k)
r2_k= r2_score(ytst, pred_k)
print((rmse_k))
mae_k= mean_absolute_error(ytst, pred_k)
model_performance["KNN Regression"] = [r2_k,rmse_k,mae_k]



# DecisionTreeRegressor
dt = DecisionTreeRegressor(min_samples_split=45, min_samples_leaf=45, random_state = 10)
dt.fit(X,y)
pred_dtr=dt.predict(Xtst)
mse_dtr= mean_squared_error(ytst,pred_dtr)
rmse_dtr=np.sqrt(mse_dtr)
r2_dtr= r2_score(ytst, pred_dtr)
print((rmse_dtr))
mae_dtr= mean_absolute_error(ytst, pred_dtr)
model_performance["Decision Tree"] = [r2_dtr,rmse_dtr,mae_dtr]


# RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
rf.fit(X, y)
pred_rfr=rf.predict(Xtst)
mse_rfr= mean_squared_error(ytst,pred_rfr)
rmse_rfr=np.sqrt(mse_rfr)
r2_rfr= r2_score(ytst, pred_rfr)
print((rmse_rfr))
mae_rfr= mean_absolute_error(ytst, pred_rfr)
model_performance["Random Forest"] = [r2_rfr,rmse_rfr,mae_rfr]


# Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=70, max_depth=5)
gbr.fit(X, y)
pred_gbr=gbr.predict(Xtst)
mse_gbr= mean_squared_error(ytst,pred_gbr)
rmse_gbr=np.sqrt(mse_gbr)
r2_gbr= r2_score(ytst, pred_gbr)
print((rmse_gbr))
mae_gbr= mean_absolute_error(ytst, pred_gbr)
model_performance["Gradient Boosting"] = [r2_gbr,rmse_gbr,mae_gbr]



# In[38]:


for i in model_performance:
    print(i,model_performance[i])


# In[39]:


d=[] 
for i in model_performance:
        k=model_performance[i]
        b=[i]
        a=b+k
        print(a)
        d.append(a)
# print(d)
table2= pd.DataFrame(d,columns = ["model","r2","rmse","mae"])
table2


# In[ ]:





# # total Votes and scores

# In[40]:


X_train=tv[:len(u)*3//4]
y_train=s[:len(s)*3//4]

X_test=tv[len(u)*3//4:]
y_test=s[len(s)*3//4:]


X = np.matrix ([[1 ,l] for l in X_train])
y = np.matrix(y_train).T
# print(X)
# print(y)
Xtst=np.matrix ([[1 ,l] for l in X_test])
ytst = np.matrix(y_test).T
# print(y.shape)
# print(X.shape)
model_perf=dict()


# In[41]:


def model_diagnostics(model, pr=True):
    """
    Returns and prints the R-squared, RMSE and the MAE for a trained model
    """
    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    if pr:
        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")
    return [r2,np.sqrt(mse),mae]

# Baseline Model
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train,y_train)
model_perf["Baseline"] = model_diagnostics(baseline)

# Linear Regressor
import sklearn
model = sklearn. linear_model . LinearRegression ( fit_intercept =False)
model.fit(X,y)
pred=model.predict(Xtst)
mse = mean_squared_error(ytst,pred)
rmse=np.sqrt(mse)
r2 = r2_score(ytst, pred)
print((rmse))
mae = mean_absolute_error(ytst, pred)
model_perf["Linear Regression"] = [r2,rmse,mae]

#Lasso Regression
lasso = LassoCV(cv=30).fit(X, y)
predl=lasso.predict(Xtst)
msel = mean_squared_error(ytst,predl)
rmsel=np.sqrt(msel)
r2l = r2_score(ytst, predl)
print((rmsel))
mael = mean_absolute_error(ytst, predl)
model_perf["Lasso Regression"] = [r2l,rmsel,mael]


# Ridge Regression
ridge = RidgeCV(cv=10).fit(X,y)
pred_r=ridge.predict(Xtst)
mse_r = mean_squared_error(ytst,pred_r)
rmse_r=np.sqrt(mse_r)
r2_r = r2_score(ytst, pred_r)
print((rmse_r))
mae_r = mean_absolute_error(ytst, pred_r)
model_perf["Ridge Regression"] = [r2_r,rmse_r,mae_r]


# Elastic Net Regression
elastic_net = ElasticNetCV(cv = 30).fit(X,y)
pred_e=elastic_net.predict(Xtst)
mse_e = mean_squared_error(ytst,pred_e)
rmse_e=np.sqrt(mse_e)
r2_e= r2_score(ytst, pred_e)
print((rmse_e))
mae_e= mean_absolute_error(ytst, pred_e)
model_perf["Elastic Net Regression"] = [r2_e,rmse_e,mae_e]


# K-Nearest Neighbor Regression
knr = KNeighborsRegressor()
knr.fit(X, y)
pred_k=knr.predict(Xtst)
mse_k= mean_squared_error(ytst,pred_k)
rmse_k=np.sqrt(mse_k)
r2_k= r2_score(ytst, pred_k)
print((rmse_k))
mae_k= mean_absolute_error(ytst, pred_k)
model_perf["KNN Regression"] = [r2_k,rmse_k,mae_k]



# DecisionTreeRegressor
dt = DecisionTreeRegressor(min_samples_split=45, min_samples_leaf=45, random_state = 10)
dt.fit(X,y)
pred_dtr=dt.predict(Xtst)
mse_dtr= mean_squared_error(ytst,pred_dtr)
rmse_dtr=np.sqrt(mse_dtr)
r2_dtr= r2_score(ytst, pred_dtr)
print((rmse_dtr))
mae_dtr= mean_absolute_error(ytst, pred_dtr)
model_perf["Decision Tree"] = [r2_dtr,rmse_dtr,mae_dtr]


# RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
rf.fit(X, y)
pred_rfr=rf.predict(Xtst)
mse_rfr= mean_squared_error(ytst,pred_rfr)
rmse_rfr=np.sqrt(mse_rfr)
r2_rfr= r2_score(ytst, pred_rfr)
print((rmse_rfr))
mae_rfr= mean_absolute_error(ytst, pred_rfr)
model_perf["Random Forest"] = [r2_rfr,rmse_rfr,mae_rfr]


# Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=70, max_depth=5)
gbr.fit(X, y)
pred_gbr=gbr.predict(Xtst)
mse_gbr= mean_squared_error(ytst,pred_gbr)
rmse_gbr=np.sqrt(mse_gbr)
r2_gbr= r2_score(ytst, pred_gbr)
print((rmse_gbr))
mae_gbr= mean_absolute_error(ytst, pred_gbr)
model_perf["Gradient Boosting"] = [r2_gbr,rmse_gbr,mae_gbr]



# In[42]:


for i in model_perf:
    print(i,model_perf[i])


# In[43]:


d=[] 
for i in model_perf:
        k=model_perf[i]
        b=[i]
        a=b+k
        print(a)
        d.append(a)
# print(d)
table3= pd.DataFrame(d,columns = ["model","r2","rmse","mae"])
table3


# # upvotes and comments

# In[44]:


X_train=u[:len(u)*3//4]
y_train=c[:len(c)*3//4]

X_test=u[len(u)*3//4:]
y_test=c[len(c)*3//4:]


X = np.matrix ([[1 ,l] for l in X_train])
y = np.matrix(y_train).T
# print(X)
# print(y)
Xtst=np.matrix ([[1 ,l] for l in X_test])
ytst = np.matrix(y_test).T
# print(y.shape)
# print(X.shape)
model_u_c=dict()


# In[45]:


def model_diagnostics(model, pr=True):
    """
    Returns and prints the R-squared, RMSE and the MAE for a trained model
    """
    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    if pr:
        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")
    return [r2,np.sqrt(mse),mae]

# Baseline Model
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train,y_train)
model_u_c["Baseline"] = model_diagnostics(baseline)

# Linear Regressor
import sklearn
model = sklearn. linear_model . LinearRegression ( fit_intercept =False)
model.fit(X,y)
pred=model.predict(Xtst)
mse = mean_squared_error(ytst,pred)
rmse=np.sqrt(mse)
r2 = r2_score(ytst, pred)
print((rmse))
mae = mean_absolute_error(ytst, pred)
model_u_c["Linear Regression"] = [r2,rmse,mae]

#Lasso Regression
lasso = LassoCV(cv=30).fit(X, y)
predl=lasso.predict(Xtst)
msel = mean_squared_error(ytst,predl)
rmsel=np.sqrt(msel)
r2l = r2_score(ytst, predl)
print((rmsel))
mael = mean_absolute_error(ytst, predl)
model_u_c["Lasso Regression"] = [r2l,rmsel,mael]


# Ridge Regression
ridge = RidgeCV(cv=10).fit(X,y)
pred_r=ridge.predict(Xtst)
mse_r = mean_squared_error(ytst,pred_r)
rmse_r=np.sqrt(mse_r)
r2_r = r2_score(ytst, pred_r)
print((rmse_r))
mae_r = mean_absolute_error(ytst, pred_r)
model_u_c["Ridge Regression"] = [r2_r,rmse_r,mae_r]


# Elastic Net Regression
elastic_net = ElasticNetCV(cv = 30).fit(X,y)
pred_e=elastic_net.predict(Xtst)
mse_e = mean_squared_error(ytst,pred_e)
rmse_e=np.sqrt(mse_e)
r2_e= r2_score(ytst, pred_e)
print((rmse_e))
mae_e= mean_absolute_error(ytst, pred_e)
model_u_c["Elastic Net Regression"] = [r2_e,rmse_e,mae_e]


# K-Nearest Neighbor Regression
knr = KNeighborsRegressor()
knr.fit(X, y)
pred_k=knr.predict(Xtst)
mse_k= mean_squared_error(ytst,pred_k)
rmse_k=np.sqrt(mse_k)
r2_k= r2_score(ytst, pred_k)
print((rmse_k))
mae_k= mean_absolute_error(ytst, pred_k)
model_u_c["KNN Regression"] = [r2_k,rmse_k,mae_k]



# DecisionTreeRegressor
dt = DecisionTreeRegressor(min_samples_split=45, min_samples_leaf=45, random_state = 10)
dt.fit(X,y)
pred_dtr=dt.predict(Xtst)
mse_dtr= mean_squared_error(ytst,pred_dtr)
rmse_dtr=np.sqrt(mse_dtr)
r2_dtr= r2_score(ytst, pred_dtr)
print((rmse_dtr))
mae_dtr= mean_absolute_error(ytst, pred_dtr)
model_u_c["Decision Tree"] = [r2_dtr,rmse_dtr,mae_dtr]


# RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
rf.fit(X, y)
pred_rfr=rf.predict(Xtst)
mse_rfr= mean_squared_error(ytst,pred_rfr)
rmse_rfr=np.sqrt(mse_rfr)
r2_rfr= r2_score(ytst, pred_rfr)
print((rmse_rfr))
mae_rfr= mean_absolute_error(ytst, pred_rfr)
model_u_c["Random Forest"] = [r2_rfr,rmse_rfr,mae_rfr]


# Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=70, max_depth=5)
gbr.fit(X, y)
pred_gbr=gbr.predict(Xtst)
mse_gbr= mean_squared_error(ytst,pred_gbr)
rmse_gbr=np.sqrt(mse_gbr)
r2_gbr= r2_score(ytst, pred_gbr)
print((rmse_gbr))
mae_gbr= mean_absolute_error(ytst, pred_gbr)
model_u_c["Gradient Boosting"] = [r2_gbr,rmse_gbr,mae_gbr]



# In[46]:


for i in model_u_c:
    print(i,model_u_c[i])


# In[47]:


d=[] 
for i in model_u_c:
        k=model_u_c[i]
        b=[i]
        a=b+k
        print(a)
        d.append(a)
# print(d)
table4= pd.DataFrame(d,columns = ["model","r2","rmse","mae"])
table4


# In[ ]:





# In[ ]:





# In[76]:


# df = pd.DataFrame(rr, columns=['total_votes','number_of_upvotes','number_of_downvotes','score','number_of_comments'])
# # print(dataframe)
# print(df['total_votes'][1:])
# df['total_votes'] = pd.to_numeric(df['total_votes'][1:])
# # form correlation matrix
# matrix = dataframe.corr()
# print(matrix)


# # Graph

# In[67]:


def model_comparison(model_performance_dict, sort_by = 'RMSE', metric = 'RMSE'):

    Rsq_list = []
    RMSE_list = []
    MAE_list = []
    for key in model_performance_dict.keys():
        Rsq_list.append(model_performance_dict[key][0])
        RMSE_list.append(model_performance_dict[key][1])
        MAE_list.append(model_performance_dict[key][2])

    props = pd.DataFrame([])

    props["R-squared"] = Rsq_list
    props["RMSE"] = RMSE_list
    props["MAE"] = MAE_list
    props.index = model_performance_dict.keys()
    props = props.sort_values(by = sort_by)

    fig, ax = plt.subplots(figsize = (12,6))

    ax.bar(props.index, props[metric], color="blue")
    plt.title(metric)
    plt.xlabel('Model')
    plt.xticks(rotation = 45)
    plt.ylabel(metric)


# In[68]:


model_comparison(model_performance_dict, sort_by = 'R-squared', metric = 'R-squared')###u vs s


# In[74]:


# def model_comp(model_performance_dict, sort_by = 'RMSE', metric = 'RMSE'):

#     Rsq_list = []
#     RMSE_list = []
#     MAE_list = []
#     for key in model_performance_dict.keys():
#         Rsq_list.append(model_performance_dict[key][0])
#         RMSE_list.append(model_performance_dict[key][1])
#         MAE_list.append(model_performance_dict[key][2])

#     props = pd.DataFrame([])

#     props["R-squared"] = Rsq_list
#     props["RMSE"] = RMSE_list
#     props["MAE"] = MAE_list
#     props.index = model_performance_dict.keys()
#     props = props.sort_values(by = sort_by)

#     fig, ax = plt.subplots(figsize = (12,6))

#     ax.plot(props.index, props[metric], color="blue")
#     ax.plot()
# #     plt.plot(props.index)
#     plt.title(metric)
#     plt.xlabel('Model')
#     plt.xticks(rotation = 45)
#     plt.ylabel(metric)


# In[75]:


# Rsq_list = []
# for key in model_performance_dict.keys():
#     Rsq_list.append(model_performance[key][0])
# plt.plot(model_performance_dict.keys(),Rsq_list, color="blue")

# ################################################
# Rsq_list = []
# for key in model_performance.keys():
#     Rsq_list.append(model_performance[key][0])
# plt.plot(Rsq_list, color="red")
# plt.plot(props["R-squared"], color="red")
# plt.show()
# ################################################
# # ax.plot()
# #     plt.plot(props.index)
# plt.title("R-squared")
# plt.xlabel('Model')
# # plt.xticks(rotation = 45)
# plt.ylabel("R-squared")


# In[ ]:





# In[72]:


model_comparison(model_performance, sort_by = 'R-squared', metric = 'R-squared')###tv vs c


# In[73]:


model_comparison(model_perf, sort_by = 'R-squared', metric = 'R-squared')###tv vs s


# In[53]:


model_comparison(model_u_c, sort_by = 'R-squared', metric = 'R-squared')###u vs c

