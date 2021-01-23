#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stat


# In[6]:


data=pd.read_csv('titanic.csv',usecols=['Age','Fare','Survived'])


# In[22]:


def impute_na(data, variable):
    df = data.copy()
    df[variable+'_random'] = df[variable]
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    
    return df[variable+'_random']


# In[23]:



data['Age']=impute_na(data,'Age')


# In[30]:


def q_q_hist(data,variable):
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    data[variable].hist()
    plt.subplot(1,2,2)
    stat.probplot(data[variable],dist="norm", plot=plt)
    plt.show()


# In[32]:


q_q_hist(data,'Fare')


# In[34]:


# log tranformation
data['log_fare']=np.log(data['Fare']+1)
q_q_hist(data,'log_fare')


# In[35]:


#reciprocal tranformation
data['Rec_Fare']=1/(data['Fare']+1)
q_q_hist(data,'Rec_Fare')


# In[39]:


#square_root
data['sqr_Fare']=data['Fare']**(1/2)
q_q_hist(data,'sqr_Fare')


# In[38]:


#Exponential 
data['Exp_Fare']=data['Fare']**(1/5)
q_q_hist(data,'sqr_Fare')


# In[40]:


#box_cox
data['Fare_box_cox'],param=stat.boxcox(data.Fare+1)
q_q_hist(data,'Fare_box_cox')


# In[ ]:




