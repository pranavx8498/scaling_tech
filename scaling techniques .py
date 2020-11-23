#!/usr/bin/env python
# coding: utf-8

# scaling required when we using concepts of eudiean distance or gradient desent 
# scaling in not used desion tree's 

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('netflix.csv')


# In[3]:


df.drop('date_added',axis=1,inplace=True)


# In[4]:


df.director.fillna('Not Mention',inplace=True)


# In[5]:


most_cast=df.cast.mode()[0]


# In[6]:


df.cast.fillna(most_cast,inplace=True)


# In[7]:


df.rating.fillna(method='pad',inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


df.head()


# In[41]:


df=pd.read_csv('data.csv')


# In[42]:


df.RM.fillna(method='pad',inplace=True)


# In[43]:


df.head()


# In[44]:


df.drop('CHAS',axis=1,inplace=True)


# In[45]:


df.isnull().sum()


# In[63]:


test=df


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x=df.iloc[:,0:-1]


# In[17]:


x.shape


# In[ ]:





# In[18]:


y=df.iloc[:,-1:]


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=2)


# In[20]:


medv=LinearRegression()


# In[21]:


medv.fit(x_train,y_train)


# In[22]:


c=medv.predict(x_test)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x,y)


# In[24]:


df_sc=sc.fit_transform(df)
dff=pd.DataFrame(df_sc,columns=df.columns)


# In[25]:


x=dff.iloc[:,0:-1]


# In[26]:


y=dff.iloc[:,-1:]


# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=2)


# In[28]:


medv=LinearRegression()


# In[29]:


medv.fit(x_train,y_train)


# In[30]:


dff.head(104)


# # standazation means centaring all values to zero
# z=(x-x_mean)/s.d. it bring it to zero

# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=2)


# In[32]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()#
sc.fit(x,y)


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


kk=LinearRegression()


# In[35]:


kk.fit(x,y)


# In[36]:


kk.score(x_test,y_test)


# # MIN_MAX  SCALING IT CONVERT DATA IN PERTUCULAR RANGE

# In[59]:


from sklearn.model_selection import train_test_split
x=test.iloc[:,:-1]
y=test.iloc[:,-1:]
from sklearn.preprocessing import MinMaxScaler


# In[60]:


mn=MinMaxScaler(feature_range=(0,20))
mn.fit(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=10)


# In[61]:


kk.fit(x,y)


# In[62]:


kk.score(x_test,y_test)


# # ROBUST SCLAR
# Z=(X-X_MEAN)/IQR   IQR=75-25 OF ALL THE DATA QUNTILE

# In[70]:


from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
rs_df=pd.DataFrame(rs.fit_transform(df),columns=df.columns)
from sklearn.model_selection import train_test_split
x=rs_df.iloc[:,:-1]
y=rs_df.iloc[:,-1:]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=10)
kk.fit(x,y)
kk.score(x_test,y_test)


# In[ ]:




