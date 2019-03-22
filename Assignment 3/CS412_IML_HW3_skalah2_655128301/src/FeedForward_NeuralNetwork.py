
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[14]:


g_random_low = 0
g_random_high = 0.01
cols = []
for i in range(16):
    cols.append('col_'+str(i+1))


# In[3]:


def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)    

def shuffleTwoArrays(X,Y):
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]


# In[4]:


def vertical_sum(row):
    ret = []
    for i in range(1,17):
        cur_sum = 0
        for j in range(16):
            cur_sum += row[(i + 16*j)]
        ret.append(cur_sum/16)
    return ret

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


# In[5]:


class NeuralNetworks:
    def __init__(self,architecture):
        self.n_layers = len(architecture)
        self.architecture = architecture
        
        self.biases = [np.random.uniform(size=(n_neurons,1),
                                         high=g_random_high,
                                         low=g_random_low)
                       for n_neurons in architecture[1:]]
        
        
        self.weights = [np.random.uniform(size=(n_neurons2,n_neurons1),
                                         high=g_random_high,
                                         low=-g_random_low)
                        for n_neurons1, n_neurons2 in zip(architecture[:-1], architecture[1:])]
        
    def resetDelta(self):
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        return delta_biases,delta_weights
    
    def addDelta(self,Delta,delta):
        return [D+d for D, d in zip(Delta, delta)]
        
    def feedForward_one_datapoint(self,x):
        x = np.reshape(x, (-1, 1))
        cur_act = x
        activations = [cur_act]
        z_values = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, cur_act)+b
            z_values.append(z)
            cur_act = sigmoid(z)
            activations.append(cur_act)
        return z_values,activations
    
    def backProp_one_datapoint(self,z_values,activations,y):
        delta_biases,delta_weights = self.resetDelta()
        
        cur_delta = (activations[-1] - y) * sigmoid_derivative(z_values[-1])
        delta_biases[-1] = cur_delta
        delta_weights[-1] = cur_delta.dot(activations[-2].T)
        
        for layer in range(2,self.n_layers):
            cur_delta = self.weights[-layer+1].T.dot(cur_delta) * sigmoid_derivative(z_values[-layer])
            delta_biases[-layer] = cur_delta
            delta_weights[-layer] = cur_delta.dot(activations[-layer-1].transpose())
        return delta_biases,delta_weights

    def getError(self,X,Y):
        DELTA_biases = [np.zeros(b.shape) for b in self.biases]
        DELTA_weights = [np.zeros(w.shape) for w in self.weights]
        for x,y in zip(X,Y):
            
            z_values,activations = self.feedForward_one_datapoint(x)
            delta_biases,delta_weights = self.backProp_one_datapoint(z_values,activations,y)
            DELTA_biases = self.addDelta(DELTA_biases,delta_biases)
            DELTA_weights = self.addDelta(DELTA_weights,delta_weights)
        return DELTA_biases,DELTA_weights
    
    def gradientDescent(self,X,Y,alpha):
        m = len(X)
        DELTA_biases,DELTA_weights = self.getError(X,Y)
        self.biases = [bias - (alpha/m)*error for bias,error in zip(self.biases,DELTA_biases)]
        self.weights = [theta - (alpha/m)*error for theta,error in zip(self.weights,DELTA_weights)]
        
    def train(self,X,Y,alpha,maxIterations):
        for i in range(maxIterations):
#             print('Iteration ',i+1)
            X,Y = shuffleTwoArrays(X,Y)
            self.gradientDescent(X,Y,alpha)
    
    def predict(self,X):
        Y = []
        for x in X:
            h = self.feedForward_one_datapoint(x)[1][-1][0][0]
            if(h < 0.5):
                Y.append(0)
            else:
                Y.append(1)

        return Y


# In[15]:


data = pd.read_csv('../input/data.csv',header=None,sep=' ')
data = data.drop([257],axis=1)
data = data.rename({0:'num'},axis=1)
data = data.loc[data.num.isin([1.0,5.0])]
data.num = data['num'].replace({1.0: 0.0, 5.0: 1.0})
data = data.sample(frac=1,random_state=1).reset_index(drop=True).astype(np.float128)
df = data.loc[0:int(0.8*len(data))-1].reset_index(drop=True)
df_test = data.loc[int(0.8*len(data)):].reset_index(drop=True)
len(data),len(df),len(df_test)


# In[16]:


col_vals = df.iloc[:,1:257].apply(vertical_sum,axis=1)
df[cols] = pd.DataFrame(col_vals.values.tolist(), columns=cols)
df['variance'] = normalize(df.iloc[:,257:273].var(axis=1),-1,1)
df['kurtosis'] = normalize(df.iloc[:,257:273].kurtosis(axis=1),-1,1)
x = df[['kurtosis','variance']]
y = df.num


# In[17]:


nn = NeuralNetworks([2,10,10,10,1])
nn.train(X=np.asarray(x),
         Y=np.asarray(y),
         alpha=0.001,
         maxIterations=100)


# In[18]:


col_vals = df_test.iloc[:,1:257].apply(vertical_sum,axis=1)
df_test[cols] = pd.DataFrame(col_vals.values.tolist(), columns=cols)
df_test['variance'] = normalize(df_test.iloc[:,257:273].var(axis=1),-1,1)
df_test['kurtosis'] = normalize(df_test.iloc[:,257:273].kurtosis(axis=1),-1,1)
x = df_test[['kurtosis','variance']]
y = df_test.num
nn.predict(np.asarray(x))

