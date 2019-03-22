
# coding: utf-8

# ## Imports

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# ## Helper Methods

# In[2]:


def vertical_sum(row):
    ret = []
    for i in range(1,17):
        cur_sum = 0
        for j in range(16):
            cur_sum += row[(i + 16*j)]
        ret.append(cur_sum/16)
    return ret


# In[3]:


def normalize(arr,a,b):
    mn = min(arr)
    mx = max(arr)
    norm_arr = []
    for x in arr:
        cur = x - mn
        cur *= (b - a)
        cur /= mx - mn
        cur += a
        norm_arr.append(cur)
    return norm_arr


# In[4]:


cols = []
for i in range(16):
    cols.append('col_'+str(i+1))


# ## Data Import and Train-Test Split

# In[5]:


data = pd.read_csv('../data/data.csv',header=None,sep=' ')
data = data.drop([257],axis=1)
data = data.rename({0:'num'},axis=1)
data = data.loc[data.num.isin([1.0,5.0])]
data = data.sample(frac=1,random_state=1).reset_index(drop=True)
df = data.loc[0:int(0.2*len(data))-1].reset_index(drop=True)
df_test = data.loc[int(0.2*len(data)):].reset_index(drop=True)
len(df),len(df_test),len(data)


# ## Data Manipulation and Feature Extraction

# In[6]:


col_vals = df.iloc[:,1:257].apply(vertical_sum,axis=1)
df[cols] = pd.DataFrame(col_vals.values.tolist(), columns=cols)
df['variance'] = normalize(df.iloc[:,257:273].var(axis=1),-1,1)
df['kurtosis'] = normalize(df.iloc[:,257:273].kurtosis(axis=1),-1,1)
df.head()


# ## First Graph 2D Scatter Plot

# In[7]:


scatter_x = np.array(df['variance'])
scatter_y = np.array(df['kurtosis'])
group = np.array(df.num)
cdict = {1.0: 'red', 5.0: 'blue'}
fig, ax = plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(10)
ax.set_title('Figure 1.1: Variance - Kurtosis Vertical Data Scatter Plot',fontsize=20)

for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 50)

plt.xlabel('Variance', fontsize=15)
plt.ylabel('Kurtosis', fontsize=15)
ax.legend()
#plt.show()
fig.savefig('../figures/figure-1_1',dpi=300)


# ## 1-NN Model - Euclidean Distance - 2D and 256D - Cross Validation Score

# In[8]:


model_knn = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
X_2f = df[['kurtosis','variance']]
y_2f = df.num

X_256f = df.iloc[:,1:257]
y_256f = df.num

cv_2f = sum(cross_val_score(model_knn, X_2f, y_2f, cv=10,scoring='f1_weighted'))/10
cv_256f = sum(cross_val_score(model_knn, X_256f, y_256f, cv=10,scoring='f1_weighted'))/10

print('Cross Validation Score 2 Features(k=10) Euclidean = ',cv_2f)
print('Cross Validation Score 256 Features (k=10) Euclidean= ',cv_256f)


# ## 1-NN Model - Manhattan Distance - 2D and 256D - Cross Validation Score

# In[9]:


model_knn = KNeighborsClassifier(n_neighbors=1,metric='manhattan')

X_2f = df[['kurtosis','variance']]
y_2f = df.num

X_256f = df.iloc[:,1:257]
y_256f = df.num

cv_2f = sum(cross_val_score(model_knn, X_2f, y_2f, cv=10,scoring='f1_weighted'))/10
cv_256f = sum(cross_val_score(model_knn, X_256f, y_256f, cv=10,scoring='f1_weighted'))/10

print('Cross Validation Score 2 Features(k=10) Manhattan = ',cv_2f)
print('Cross Validation Score 256 Features (k=10) Manhattan= ',cv_256f)


# ## 1-NN Model - Chebyshev Distance - 2D and 256D - Cross Validation Score

# In[10]:


model_knn = KNeighborsClassifier(n_neighbors=1,metric='chebyshev')

X_2f = df[['kurtosis','variance']]
y_2f = df.num

X_256f = df.iloc[:,1:257]
y_256f = df.num

cv_2f = sum(cross_val_score(model_knn, X_2f, y_2f, cv=10,scoring='f1_weighted'))/10
cv_256f = sum(cross_val_score(model_knn, X_256f, y_256f, cv=10,scoring='f1_weighted'))/10

print('Cross Validation Score 2 Features(k=10) Chebyshev = ',cv_2f)
print('Cross Validation Score 256 Features (k=10) Chebyshev= ',cv_256f)


# ## 1-NN Model - Euclidean Distance - 2D  - [-1,1] Square Region Color Separation

# In[11]:


model_knn = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
model_knn = model_knn.fit(X=X_2f,y=y_2f)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        cPred.append(model_knn.predict([[xP,yP]])[0])
        
scatter_x = np.array(xPred)
scatter_y = np.array(yPred)
group = np.array(cPred)
cdict = {1.0: 'red', 5.0: 'blue'}
fig, ax = plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(10)
ax.set_title('Figure 1.2: 2D 1-Nearest Neighbor Scatter Plot',fontsize=20)
plt.xlabel('Variance', fontsize=15)
plt.ylabel('Kurtosis', fontsize=15)

for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 1)

ax.legend()
#plt.show()
fig.savefig('../figures/figure-1_2',dpi=300)


# ## k-NN Model - Euclidean Distance - 256D  - 95% Confidence Interval CV Error

# In[12]:


k = []
cv_score = []
df_cv = pd.DataFrame()
X_256f = df.iloc[:,1:257]
y_256f = df.num


for i in range(1,50):
    if(i%2 is 0):
        continue
    model_knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    cv_score.append(cross_val_score(model_knn, X_256f, y_256f, cv=10,scoring='f1_weighted'))
    k.append(i)
    
df_cv['cv_score'] = cv_score
df_cv['k'] = k

mean = df_cv['cv_score'].apply(np.mean)
std = df_cv['cv_score'].apply(np.std)
z = mean - 2*std
zmax = np.max(z)
mask = np.array(z) == zmax
color = np.where(mask, 'red', 'blue')

plt.figure(figsize=(16, 10))
plt.xlabel('Number of Nearest Neighbors',fontsize=15)
plt.ylabel('10-Fold Cross Validated F1 Score with 95% Confidence Interval',fontsize=15)
plt.title('Figure 1.3: k versus Ecv Plot of 256D k-NN Model',fontsize=20)
plt.errorbar(k, mean, xerr=0.2, yerr=2*std, linestyle='',color=color)
plt.savefig('../figures/figure-1_3',dpi=300)
#plt.show()


# ## k-NN Model - Euclidean Distance - 2D  - 95% Confidence Interval CV Error

# In[13]:


k = []
cv_score = []
df_cv = pd.DataFrame()
X_2f = df[['kurtosis','variance']]
y_2f = df.num

for i in range(1,50):
    if(i%2 is 0):
        continue
    model_knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    cv_score.append(cross_val_score(model_knn, X_2f, y_2f, cv=10,scoring='f1_weighted'))
    k.append(i)
    
df_cv['cv_score'] = cv_score
df_cv['k'] = k

mean = df_cv['cv_score'].apply(np.mean)
std = df_cv['cv_score'].apply(np.std)
z = mean - 2*std
zmax = np.max(z)
mask = np.array(z) == zmax
color = np.where(mask, 'red', 'blue')

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Number of Nearest Neighbors',fontsize=15)
plt.ylabel('10-Fold Cross Validated F1 Score with 95% Confidence Interval',fontsize=15)
plt.title('Figure 1.5: k versus Ecv Plot of 2D k-NN Model',fontsize=20)
plt.errorbar(k, mean,xerr=0.2, yerr=2*std, linestyle='',fmt='o')
plt.savefig('../figures/figure-1_5',dpi=300)
#plt.show()


# ## Extra Credit - IRIS Dataset

# In[14]:


#Importing Iris Data and Normalizing
df = pd.read_csv('../data/iris.data',header=None,sep=',')
df.columns = ['s_l','s_w','p_l','p_w','fl']
df['s_l'] = normalize(df['s_l'],0,1)
df['s_w'] = normalize(df['s_w'],0,1)
df['p_l'] = normalize(df['p_l'],0,1)
df['p_w'] = normalize(df['p_w'],0,1)
df.head()


# In[15]:


#knn model for multiple k's SL and SW
k = []
cv_score = []
df_cv = pd.DataFrame()
X_2f = df[['s_l','s_w']]
y_2f = df.fl

for i in range(1,50):
    if(i%2 is 0):
        continue
    model_knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    cv_score.append(cross_val_score(model_knn, X_2f, y_2f, cv=10,scoring='f1_weighted'))
    k.append(i)
    
df_cv['cv_score'] = cv_score
df_cv['k'] = k

mean = df_cv['cv_score'].apply(np.mean)
std = df_cv['cv_score'].apply(np.std)
z = mean - 2*std
zmax = np.max(z)
mask = np.array(z) == zmax
color = np.where(mask, 'red', 'blue')

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Number of Nearest Neighbors',fontsize=15)
plt.ylabel('10-Fold Cross Validated F1 Score with 95% Confidence Interval',fontsize=15)
plt.title('Figure 1.6: k versus Ecv Plot of k-NN Model on Sepal Length and Sepal Width',fontsize=20)
plt.errorbar(k, mean,xerr=0.2, yerr=2*std, linestyle='',color=color)
plt.savefig('../figures/figure-1_6',dpi=300)
print('Best Model\'s F1 Score:',mean[list(z).index(zmax)])
#plt.show()


# In[16]:


#knn model for multiple k's PL and PW
k = []
cv_score = []
df_cv = pd.DataFrame()
X_2f = df[['p_l','p_w']]
y_2f = df.fl

for i in range(1,50):
    if(i%2 is 0):
        continue
    model_knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    cv_score.append(cross_val_score(model_knn, X_2f, y_2f, cv=10,scoring='f1_weighted'))
    k.append(i)
    
df_cv['cv_score'] = cv_score
df_cv['k'] = k

mean = df_cv['cv_score'].apply(np.mean)
std = df_cv['cv_score'].apply(np.std)
z = mean - 2*std
zmax = np.max(z)
mask = np.array(z) == zmax
color = np.where(mask, 'red', 'blue')

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Number of Nearest Neighbors',fontsize=15)
plt.ylabel('10-Fold Cross Validated F1 Score with 95% Confidence Interval',fontsize=15)
plt.title('Figure 1.7: k versus Ecv Plot of k-NN Model on Petal Length and Petal Width ',fontsize=20)
plt.errorbar(k, mean,xerr=0.2, yerr=2*std, linestyle='',color=color)
plt.savefig('../figures/figure-1_7',dpi=300)
print('Best Model\'s F1 Score:',mean[list(z).index(zmax)])
#plt.show()


# In[17]:


#Scatter Plot of Data in 2D SL and SW

scatter_x = np.array(df['s_l'])
scatter_y = np.array(df['s_w'])
group = np.array(df['fl'])
cdict = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
fig, ax = plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(10)
ax.set_title('Figure 1.8: Sepal Length - Sepal Width Iris Scatter Plot',fontsize=20)

for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 50)

plt.xlabel('Sepal Length', fontsize=15)
plt.ylabel('Sepal Width', fontsize=15)
ax.legend()
#plt.show()
fig.savefig('../figures/figure-1_8',dpi=300)


# In[18]:


#Scatter Plot of Data in 2D SL and SW

scatter_x = np.array(df['p_l'])
scatter_y = np.array(df['p_w'])
group = np.array(df['fl'])
cdict = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
fig, ax = plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(10)
ax.set_title('Figure 1.9: Petal Length - Petal Width Iris Scatter Plot',fontsize=20)

for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 50)

plt.xlabel('Petal Length', fontsize=15)
plt.ylabel('Petal Width', fontsize=15)
ax.legend()
#plt.show()
fig.savefig('../figures/figure-1_9',dpi=300)


# In[19]:


#knn model for multiple k's All features
k = []
cv_score = []
df_cv = pd.DataFrame()
X_2f = df[['p_l','p_w','s_l','s_w']]
y_2f = df.fl

for i in range(1,50):
    if(i%2 is 0):
        continue
    model_knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    cv_score.append(cross_val_score(model_knn, X_2f, y_2f, cv=10,scoring='f1_weighted'))
    k.append(i)
    
df_cv['cv_score'] = cv_score
df_cv['k'] = k

mean = df_cv['cv_score'].apply(np.mean)
std = df_cv['cv_score'].apply(np.std)
z = mean - 2*std
zmax = np.max(z)
mask = np.array(z) == zmax
color = np.where(mask, 'red', 'blue')

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Number of Nearest Neighbors',fontsize=15)
plt.ylabel('10-Fold Cross Validated F1 Score with 95% Confidence Interval',fontsize=15)
plt.title('Figure 1.10: k versus Ecv Plot of k-NN Model on All Features of Iris Data ',fontsize=20)
plt.errorbar(k, mean,xerr=0.2, yerr=2*std, linestyle='',color=color)
plt.savefig('../figures/figure-1_10',dpi=300)
print('Best Model\'s F1 Score:',mean[list(z).index(zmax)])
#plt.show()


# ## Extra Credit - Zoo Data

# In[20]:


#Data Import and Normalization
df = pd.read_csv('../data/zoo.data',header=None,sep=',')
df = df[df[17].isin([1,2,4,7])]
for col in df.iloc[:,1:17].columns:
    df[col] = pd.to_numeric(df[col])
    df[col] = normalize(df[col],0,1)
df.head()


# In[21]:


# CV Plot for All Features
k = []
cv_score = []
df_cv = pd.DataFrame()
    
X_2f = np.asarray(df.iloc[:,1:17])
y_2f = np.asarray(df[17])

for i in range(1,50):
    if(i%2 is 0):
        continue
    model_knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    cur_cv = cross_val_score(model_knn, X_2f, y_2f, cv=5,scoring='accuracy')
    cv_score.append(cur_cv)
    k.append(i)
    
df_cv['cv_score'] = cv_score
df_cv['k'] = k

mean = df_cv['cv_score'].apply(np.mean)
std = df_cv['cv_score'].apply(np.std)
z = mean - 2*std
zmax = np.max(z)
mask = np.array(z) == zmax
color = np.where(mask, 'red', 'blue')

plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Number of Nearest Neighbors',fontsize=15)
plt.ylabel('10-Fold Cross Validated F1 Score with 95% Confidence Interval',fontsize=15)
plt.title('Figure 1.11: k versus Ecv Plot of k-NN Model on All Features of Zoo Data',fontsize=20)
plt.errorbar(k, mean,xerr=0.2, yerr=2*std, linestyle='',color=color)
plt.savefig('../figures/figure-1_11',dpi=300)
print('Best Model\'s F1 Score:',mean[list(z).index(zmax)])
#plt.show()

