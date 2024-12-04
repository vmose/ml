#!/usr/bin/env python
# coding: utf-8

# In[101]:


#using correlation to make a recommendation system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


column_names =['user_id','item_id','rating','timestamp']


# In[10]:


df=pd.read_csv(r'c:\Input\u.data',sep='\t',names=column_names)
df.head()


# In[12]:


movie_titles= pd.read_csv(r'c:\Input\Movie_Id_Titles')
movie_titles.head()


# In[26]:


df=pd.merge(df,movie_titles,on='item_id').sort_values('item_id')
df.head()


# In[36]:


df.groupby('title').mean()['rating'].sort_values(ascending=False).head()


# In[40]:


df.groupby('title').count()['rating'].sort_values(ascending=False).head()


# In[43]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
ratings.head()


# In[44]:


ratings['num_of_ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# In[45]:


ratings


# In[47]:


ratings['rating'].hist(bins=70)


# In[48]:


ratings['num_of_ratings'].hist(bins=70)


# In[55]:


plt.figure(figsize=(8,6))
sb.jointplot(ratings['rating'],ratings['num_of_ratings'],alpha=0.5)


# In[57]:


matrix1=df.pivot_table(index='user_id',columns='title',values='rating')
matrix1.head()


# In[65]:


ratings.sort_values('num_of_ratings',ascending=False).head()


# In[75]:


mat_starwars=matrix1['Star Wars (1977)']


# In[78]:


similar2starwars=pd.DataFrame(matrix1.corrwith(mat_starwars).sort_values(ascending=False),columns=['Correlation'])


# In[80]:


similar2starwars.head(10)


# In[81]:


similar2starwars=similar2starwars.join(ratings['num_of_ratings'])


# In[102]:


#essentially, if you like starwars(1977) youre likely to like these 5
similar2starwars[similar2starwars['num_of_ratings']>100].head(5)


# In[92]:


mat_fargo=matrix1['Fargo (1996)']


# In[95]:


similar2fargo=pd.DataFrame(matrix1.corrwith(mat_fargo).sort_values(ascending=False),columns=['Correlation'])


# In[97]:


similar2fargo=similar2fargo.join(ratings['num_of_ratings'])


# In[103]:


#same process but with Fargo
similar2fargo[similar2fargo['num_of_ratings']>100].head(5)


# In[ ]:




