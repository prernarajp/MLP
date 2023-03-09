#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd


# In[49]:


data=pd.read_csv("iris1.csv")
data


# In[50]:


data['variety'].value_counts()


# In[66]:


x=data.iloc[:,0:4].values
x


# In[68]:


y=data.iloc[:,4].values
y


# In[69]:


from sklearn.preprocessing import LabelEncoder


# In[70]:


labelencoder_y=LabelEncoder()


# In[71]:


y=labelencoder_y.fit_transform(y)
y


# In[72]:


from sklearn.model_selection import train_test_split 


# In[73]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train


# In[74]:


x_train


# In[75]:


x_test.size


# In[76]:


y_train


# In[77]:


y_test.size


# In[78]:


from sklearn.linear_model import LogisticRegression


# In[79]:


logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)


# In[81]:


y_pred=logmodel.predict(x_test)


# In[82]:


y_pred
np.sort(y_pred)


# In[83]:


y_test
np.sort(y_test)


# In[84]:


from sklearn.metrics import confusion_matrix


# In[85]:


confusion_matrix(y_test,y_pred)


# In[ ]:




