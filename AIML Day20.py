#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd


# In[24]:


Pat = [5, 4, 4, 3, 9, 4]
Jack = [4, 8, 7, 5, 1, 5]
Alex = [9, 9, 8, 10, 4, 10]


# In[25]:


all_scores = Pat+Jack+Alex
company_names = (['Pat'] * len(Pat)) +  (['Jack'] * len(Jack)) +  (['Alex'] * len(Alex))


# In[26]:


data = pd.DataFrame({'names': company_names, 'score': all_scores})

data


# In[27]:


data.groupby('names').mean()


# In[28]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[29]:


lm = ols('score ~ names',data=data).fit()
table = sm.stats.anova_lm(lm)
print(table)


# In[30]:


pip install pingouin


# In[31]:


from pingouin import ancova


# In[32]:


overall_mean = data['score'].mean()
print('overall_mean=',overall_mean)


# In[33]:


data['overall_mean'] = overall_mean
ss_total = sum((data['score'] - data['overall_mean'])**2)
print('ss_total(btwn)=',ss_total)


# In[34]:


group_means = data.groupby('names').mean()
group_means = group_means.rename(columns = {'score': 'group_mean'})
group_means # Xbar 1 , Xbar 2 , Xbar 3


# In[35]:


data = data.merge(group_means, left_on = 'names', right_index = True)
data


# In[36]:


ss_residual = sum((data['score'] - data['group_mean'])**2)
print('ss_residual (within)=',ss_residual)


# In[37]:


ss_explained = sum((data['group_mean'] - data['overall_mean_x'])**2)
print('ss_explained(between)=',ss_explained)


# In[38]:


n_groups = len(set(data['names'])) 
n_obs = data.shape[0] 
df_residual = n_obs - n_groups 
ms_residual = ss_residual / df_residual 
print('ms_residual(within)=',ms_residual)


# In[39]:


critical_value=scipy.stats.f.ppf(1-0.05,df_explained,df_residual)
critical_value


# In[40]:


import seaborn as sns 
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
print(sns.barplot(x='names',y='score',data=data))


# In[ ]:




