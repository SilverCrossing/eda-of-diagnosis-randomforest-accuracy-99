#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
df = pd.read_csv("/kaggle/input/cancer-data/Cancer_Data.csv")
df


# In[3]:


df.info()


# In[4]:


df.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
df.head()


# In[5]:


df.describe()


# In[6]:


df1 = df.copy()
df.drop("diagnosis", axis = 1, inplace = True)
df = pd.concat([df, df1["diagnosis"]], axis = 1)
df.head()


# In[7]:


df["diagnosis"].value_counts()


# # Exploratory Data Analysis

# In[8]:


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


# Distribution of all features by target(diagnosis)
plt.figure(figsize = (10, 20))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(10, 3, i)
    sns.histplot(x = df[col], hue = df["diagnosis"], multiple = "dodge")
    plt.title(f"Distribution of {col} Data")
    plt.tight_layout()
    plt.xticks(rotation = 90)
    plt.plot()


# In[10]:


# Malingn(bad) : 0 / Benign(good) : 1
df["diagnosis"] = df["diagnosis"].map({"M" : 1, "B" : 0})
df.head()


# In[11]:


# Correlation of Diagnosis
df_corr = df.corr()
df_corr["diagnosis"].sort_values(ascending = False)


# In[12]:


plt.figure(figsize = (15, 15))
sns.heatmap(df_corr, fmt = ".2f", annot = True, cmap = "YlGnBu")
plt.show()


# In[13]:


# Check skewness of all features
plt.figure(figsize = (10, 20))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(10, 3, i)
    skewness = df[col].skew()
    sns.distplot(df[col], kde = True, label = "Skew = %.3f" %(skewness))
    plt.title(f"Skewness of {col} Data")
    plt.tight_layout()
    plt.legend(loc = "best")
    plt.xticks(rotation = 90)
    plt.plot()


# ### radius_se / perimeter_se / area_se / smoothness_se / concavity_se / symmetry_se / fractal_dimension_se have many outliers.
# ### We have to remove them on each feature.

# # Preprocessing

# In[14]:


import numpy as np
from scipy import stats
from scipy.stats import shapiro
num_feat = ["radius_se", "perimeter_se", "area_se", "smoothness_se" , "concavity_se", "symmetry_se", "fractal_dimension_se"]
for col in num_feat:
    print(f"Columns : {col}")
    plt.hist(df[col], density = True, alpha = 0.6, color = 'b')
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(df[col]), np.std(df[col]))
    plt.plot(x, p, 'k--', linewidth = 2)
    plt.show()
    
    stat, p = shapiro(df[col])
    print("Statistics = %.3f, p = %.3f" %(stat, p))
    
    alpha = 0.05
    if p > alpha:
        print("Dats looks Gaussian Distribution (fail to reject H0) \n")
    else:
        print("Data does not look Gaussian Distribution (reject H0) \n")


# ### They need to be removed by IQR method.

# In[15]:


# IQR method
for col in num_feat:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5*IQR)) & (df[col] <= (Q3 + 1.5*IQR))]
df


# In[16]:


plt.figure(figsize = (10, 20))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(10, 3, i)
    skewness = df[col].skew()
    sns.distplot(df[col], kde = True, label = "Skew = %.3f" %(skewness))
    plt.title(f"Skewness of {col} Data [outliers removed")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.plot()


# ### You see skewness that many outliers are removed by IQR method.

# In[17]:


# Split dataset with train/test
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Machine Learning - Classification of diagnosis

# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[19]:


def accuracy(X_train, X_test, y_train, y_test):
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_reg = reg.predict(X_test)
    
    svc = SVC()
    svc.fit(X_train, y_train)
    y_svc = svc.predict(X_test)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_dtc = dtc.predict(X_test)
    
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_rfc = rfc.predict(X_test)
    
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_gbc = gbc.predict(X_test)
    
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_abc = abc.predict(X_test)
    
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_knc = knc.predict(X_test)
    
    return accuracy_score(y_test, y_reg), accuracy_score(y_test, y_svc), accuracy_score(y_test, y_dtc), accuracy_score(y_test, y_rfc), accuracy_score(y_test, y_gbc), accuracy_score(y_test, y_abc), accuracy_score(y_test, y_knc)


# In[20]:


def precision(X_train, X_test, y_train, y_test):
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_reg = reg.predict(X_test)
    
    svc = SVC()
    svc.fit(X_train, y_train)
    y_svc = svc.predict(X_test)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_dtc = dtc.predict(X_test)
    
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_rfc = rfc.predict(X_test)
    
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_gbc = gbc.predict(X_test)
    
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_abc = abc.predict(X_test)
    
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_knc = knc.predict(X_test)
    
    return precision_score(y_test, y_reg), precision_score(y_test, y_svc), precision_score(y_test, y_dtc), precision_score(y_test, y_rfc), precision_score(y_test, y_gbc), precision_score(y_test, y_abc), precision_score(y_test, y_knc)


# In[21]:


def recall(X_train, X_test, y_train, y_test):
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_reg = reg.predict(X_test)
    
    svc = SVC()
    svc.fit(X_train, y_train)
    y_svc = svc.predict(X_test)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_dtc = dtc.predict(X_test)
    
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_rfc = rfc.predict(X_test)
    
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_gbc = gbc.predict(X_test)
    
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_abc = abc.predict(X_test)
    
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_knc = knc.predict(X_test)
    
    return recall_score(y_test, y_reg), recall_score(y_test, y_svc), recall_score(y_test, y_dtc), recall_score(y_test, y_rfc), recall_score(y_test, y_gbc), recall_score(y_test, y_abc), recall_score(y_test, y_knc)


# In[22]:


def f1(X_train, X_test, y_train, y_test):
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_reg = reg.predict(X_test)
    
    svc = SVC()
    svc.fit(X_train, y_train)
    y_svc = svc.predict(X_test)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_dtc = dtc.predict(X_test)
    
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_rfc = rfc.predict(X_test)
    
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_gbc = gbc.predict(X_test)
    
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_abc = abc.predict(X_test)
    
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_knc = knc.predict(X_test)
    
    return f1_score(y_test, y_reg), f1_score(y_test, y_svc), f1_score(y_test, y_dtc), f1_score(y_test, y_rfc), f1_score(y_test, y_gbc), f1_score(y_test, y_abc), f1_score(y_test, y_knc)


# In[23]:


scores = pd.DataFrame(columns = ["REG", "SVC", "DTC", "RFC", "GBC", "ABC", "KNC"],
                     index = ["ACC", "PREC", "REC", "F1"])
acc = [reg, svc, dtc, rfc, gbc, abc, knc] = accuracy(X_train, X_test, y_train, y_test)
scores.iloc[0, :] = acc
prec = [reg, svc, dtc, rfc, gbc, abc, knc] = precision(X_train, X_test, y_train, y_test)
scores.iloc[1, :] = prec
rec = [reg, svc, dtc, rfc, gbc, abc, knc] = recall(X_train, X_test, y_train, y_test)
scores.iloc[2, :] = rec
f_1 = [reg, svc, dtc, rfc, gbc, abc, knc] = f1(X_train, X_test, y_train, y_test)
scores.iloc[3, :] = f_1
scores


# ### RandomForestClassifier has highest score in 7 models.
# ### Let's find Train/Test model score of RandomForestClassifier.

# In[24]:


train_scores = []
test_scores = []

for i in range(2, 10):
    rfc = RandomForestClassifier(n_estimators = i * 50, max_features = i/10, max_depth = i, min_samples_split = i)
    rfc.fit(X_train, y_train)
    train_scores.append(rfc.score(X_train, y_train))
    test_scores.append(rfc.score(X_test, y_test))
    
sns.lineplot(train_scores, marker = '*', color = 'b')
sns.lineplot(test_scores, marker = 'o', color = 'r')


# In[25]:


# i = 2
rfc = RandomForestClassifier(n_estimators = 100, max_features = 0.2, max_depth = 2, min_samples_split = 2)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("RFC Train model Score :", rfc.score(X_train, y_train))
print("RFC Test model Score :", rfc.score(X_test, y_test))
print("----------")
print("Accuracy Score of RFC :", accuracy_score(y_test, y_pred))
print("Precision Score of RFC :", precision_score(y_test, y_pred))
print("Recall Score of RFC :", recall_score(y_test, y_pred))
print("F1 Score of RFC :", f1_score(y_test, y_pred))


# In[ ]:




