#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.io import loadmat


# In[9]:


from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging


# In[10]:


from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score


# ## Define Data file and Read X & Y

# In[11]:


mat_file_list = ['arrhythmia.mat',
                 'cardio.mat',
                 'glass.mat',
                 'ionosphere.mat',
                 'letter.mat',
                 'lympho.mat',
                 'mnist.mat',
                 'musk.mat',
                 'optdigits.mat',
                 'pendigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                 'shuttle.mat',
                 'vertebral.mat',
                 'vowels.mat',
                 'wbc.mat']


# In[12]:


mat_file_list


# In[13]:


data=loadmat('C:/Users/Suhas Jain/Desktop/AIML_P1/cardio')


# In[14]:


data


# In[15]:


len(data)


# In[16]:


data.keys()


# In[17]:


data.values()


# ## Input(Independent) Feature shape in Mat file format

# In[14]:


type(data['X']),data['X'].shape


# ## Dependent/ Target/ Output Feature shape

# In[15]:


type(data['y']),data['y'].shape


# ## Exploring all Mat files

# In[18]:


from time import time
random_state = np.random.RandomState(42)

for mat_file in mat_file_list:
    print("\n....Processing", mat_file, '....')
    mat = loadmat(os.path.join('data',appendmat,datamat_file))
    
    X = mat['X']
    y = mat['y'].ravel()
    
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction *100, ndigits=4)
    
    #Construct containers for saving results
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    
    #60% data for training and 40% data for Testing
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.4, random_state=random_state)
    
    #standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)
    
    classifiers = {'Angel-Based Outliers Detectors (ABOD)': ABOD(
        contamination=outliers_fractions),
            'Cluster-Based Local Outlier Factor': CBLOF(
                   contamination=outliers_fractions, check_estimator=False, 
                   random_state=random_state),
            'Feature Bagging ': FeatureBagging (contamination=outliers_fractions,
                                               random_state=random_state),
             'Histogram-Based Outlier Detection': HBOS(
                    contamination=outliers_fractions),
             'Isolation-Forest': IForest(contamination=outliers_fractions,
                                         random_state=random_state),
             'K Nearest Neighbours (KNN)': KNN(contamination=outliers_fractions),
             'Local Outlier Factor (LOF)': LOF(
                 contamination=outliers_fractions),
             'Minimum Covariance Determinant (MCD)': MCD(
                 contamination=outliers_fractions, random_state=random_state),
             'One-Class SVM OCSVM': OCSVM(contamination=outliers_fractions),
             'Principal Component Analysis (PCA)': PCA(
                 contamination=outliers_fractions, random_state=random_state) 
    }
    
    for clf_name, clf in classifiers.items():
        t0 = time()
        clf.fit(X_train_norm)
        test_scores = clf.decisions_function(X_test_norm)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        time_list.append(duration)
        
        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
        
        print('{clf_name} ROC:{roc}, precision @ rank n:{prn},'
              'execution time: {duration}s'.format(
            clf_name=clf_name, roc=roc, prn=prn, duration=duration))
        
        roc_list.append(roc)
        prn_list.append(prn)
        
        temp_df = pd.Dataframe(time_list).transpose()
        temp_df.columns=df_columns
        time_df = pd_concat([time_df, temp_df], axis=0)
        
        temp_df = pd.Dataframe(roc_list).transpose()
        temp_df.columns=df_columns
        time_df = pd_concat([roc_df, temp_df], axis=0)
        
        temp_df = pd.Dataframe(prn_list).transpose()
        temp_df.columns=df_columns
        time_df = pd_concat([prn_df, temp_df], axis=0)


# In[ ]:





# ## Define nine outlayer detection tools to be compared

# In[19]:


df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc', 'ABOD', 'CBLOF', 'FB', 'HBOS', 'IForest', 'KNN', 
              'LOF', 'MCD', 'OCSVM', 'PCA']


roc_df = pd.DataFrame(columns=df_columns)

prn_df = pd.DataFrame(columns=df_columns)

time_df = pd.DataFrame(columns=df_columns)

