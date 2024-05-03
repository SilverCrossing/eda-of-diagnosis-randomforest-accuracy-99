#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np   # linear algebra
import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))   # 用于路径拼接文件路径，如果不存在以'/'开始的参数，则函数会自动加上

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("./Cancer_Data.csv")   # 读取csv文件
df


# In[ ]:


df.info()   # 查看df文件中的信息


# In[ ]:


df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)   # x.drop()方法删除指定行列（axis=1为删除列）
# 采用inplace=True之后，原数组名对应的内存值直接改变，也就是说原数组直接就被替换
df.head()   # x.head()取数据的前n行数据，默认是前5行，不对数据内容做任何改变。没有print语句则x.head()方法只是选择数据


# In[ ]:

# pandas有两个核心数据结构Series和DataFrame，分别对应了一维的序列和二维的表结构
df.describe()   # x.describe()方法就是返回这两个核心数据结构的统计变量


# In[ ]:


df1 = df.copy()   # 复制出一个新的表
df.drop("diagnosis", axis=1, inplace=True)

df = pd.concat([df, df1["diagnosis"]], axis=1)    # pd.concat()函数可以沿着指定的轴将多个dataframe或者series拼接到一起
# 这一点和另一个常用的pd.merge()函数不同，pd.merge()函数只能实现两个表的拼接

df.head()


# In[ ]:


# 在pandas中，value_counts常用于数据表的计数及排序，它可以用来查看数据表中，指定列里有多少个不同的数据值，并计算每个不同值有在该列中的个数
# 同时还能根据需要进行排序
df["diagnosis"].value_counts()


# # Exploratory Data Analysis

# In[ ]:


import warnings
warnings.filterwarnings('ignore')   # 忽略警告消息
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Distribution of all features by target(diagnosis)
plt.figure(figsize=(30, 25))   # 创建宽30高25的表格

# enumerate在字典上是枚举、列举的意思，也可用于列表/文件，对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串）enumerate将其组成一个索引序列
# 利用它可以同时获得索引和值，可以在for循环中得到计数
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(6, 5, i)   # 创建6行5列的子图，操作其中第i个
    # 创建直方图，x轴为df[col]，hue参数来根据df["diagnosis"]列对数据进行颜色区分，使用"dodge"模式来堆叠不同组别的直方柱
    sns.histplot(x=df[col], hue=df["diagnosis"], multiple="dodge")
    plt.title(f"Distribution of {col} Data")   # 设定表格标题
    plt.tight_layout()   # 自动调整子图参数，使其填充整个图像区域
    plt.xticks(rotation=90)   # x.xticks()获取或设置当前x轴刻度位置和标签。若不传递任何参数，则返回当前刻度值，rotation=90让文本标签逆时针旋转90度
    plt.plot()   # 画图
plt.show()


# In[ ]:


# Malingn(bad) : 0 / Benign(good) : 1
# 使用Pandas库中的map函数，将df数据框中"diagnosis"列的值从"M"和"B"映射为1和0。这通常用于将分类数据转换为数值数据，便于后续的数据分析或机器学习模型的训练
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df.head()


# In[ ]:


# Correlation of Diagnosis
df_corr = df.corr()   # x.corr()方法计算数据框df中所有列的相关系数矩阵，范围为-1到1，表示负相关到正相关

# 将数据框df_corr中"diagnosis"列的值按照降序排序。由于df_corr是一个相关系数矩阵，这个操作将会把与"diagnosis"列相关性最强的变量排在最前面
df_corr["diagnosis"].sort_values(ascending=False)   # 这样做的目的是为了快速识别哪些变量与"diagnosis"的关系最为密切


# In[ ]:


plt.figure(figsize=(15, 15))
sns.heatmap(df_corr, fmt=".2f", annot=True, cmap="OrRd")   # 创建一个热图（heatmap），用于显示df_corr数据框中的相关系数矩阵
plt.title("heatmap")
# fmt=".2f"设置热图中注释的数字格式，.2f表示保留两位小数
# annot参数设置为True表示在热图的每个单元格中添加注释，即显示相关系数的具体数值
# cmap参数指定热图的颜色映射方案，"YlGnBu"是一种颜色渐变方案，从黄色（Yellow）到绿色（Green）再到蓝色（Blue）。"OrRd"则是从橙色到红色

plt.show()


# In[ ]:


# Check skewness of all features
plt.figure(figsize=(30, 25))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(6, 5, i)
    skewness = df[col].skew()   # 计算当前列的偏度。偏度是衡量数据分布对称性的指标，偏度值越大，表示数据分布越不对称
    sns.distplot(df[col], kde=True, label="Skew = %.3f" %(skewness))   # 绘制当前列的分布图，kde=True表示绘制核密度估计，label显示偏度值
    plt.title(f"Skewness of {col} Data")
    plt.tight_layout()
    plt.legend(loc="best")   # 显示图例，位置在最佳位置
    plt.xticks(rotation=90)
    plt.plot()
plt.show()


"""
偏度（Skewness）是衡量数据分布非对称程度的统计量。它反映了数据分布相对于平均值的偏斜方向和程度。具体来说：
偏度为零：表示数据分布是对称的，即数据在平均值两侧分布均匀
偏度大于零：表示数据分布是右偏的，或称正偏态，数据的右尾部（较大值）比左尾部长
偏度小于零：表示数据分布是左偏的，或称负偏态，数据的左尾部（较小值）比右尾部长

在统计分析中，偏度的作用主要包括：
描述数据分布特征：偏度可以帮助我们了解数据集中值的分布情况，尤其是数据是否均匀分布在平均值周围
检验正态性：许多统计方法假设数据呈正态分布。偏度接近零可以作为数据近似正态分布的一个指标
影响统计推断：偏度影响数据的均值和中位数的关系，进而影响某些统计测试的结果，如t检验、方差分析等
通过计算偏度，我们可以得到数据分布形状的一个量化描述，这对于数据分析和决策制定是非常有用的。例如，如果数据高度偏斜，可能需要进行数据转换或使用非参数统计方法来进行更准确的分析

偏度是通过数据集的三阶标准化矩来计算的，它反映了数据分布的不对称性。具体的计算公式如下：
S=(1/n)*(sum((i=1, n), ((X_i - mu)/sigma)**3))
其中：
(n)是数据点的数量，(X_i)是单个数据点的值，(mu)是数据集的均值，(sigma)是数据集的标准差

计算步骤如下：
计算数据集的均值(mu)和标准差(sigma)，对于数据集中的每一个数据点(X_i)，计算(X_i) 与均值(mu)的差值，然后除以标准差(sigma)
将上一步的结果立方，然后对所有数据点的这个立方值求和。最后，将求和的结果除以数据点的总数(n)
"""

"""
核密度估计（Kernel Density Estimation，简称KDE）是一种用于估计随机变量的概率密度函数的非参数方法。它可以被看作是直方图的平滑版本
其中平滑程度由一个参数，即核（kernel）的带宽（bandwidth）控制。核密度估计的基本思想是在每个数据点处放置一个核，这个核可以是高斯核（即正态分布曲线）
然后将所有核加起来，形成一个平滑的密度估计

以下是一些关键点：
核函数：用于平滑数据点的函数，常见的有高斯核
带宽：决定核函数宽度的参数，带宽越大，曲线越平滑；带宽越小，曲线越尖锐
优点：KDE可以展示出数据的分布形态，特别是在数据分布不规则时比直方图更加清晰
应用：KDE常用于数据探索分析，可以帮助我们直观地理解数据的分布特性
"""

# ### radius_se / perimeter_se / area_se / smoothness_se / concavity_se / symmetry_se / fractal_dimension_se have many outliers.
# ### We have to remove them on each feature.

# # Preprocessing

# In[ ]:


# import numpy as np
# from scipy import stats
# from scipy.stats import shapiro
# num_feat = ["radius_se", "perimeter_se", "area_se", "smoothness_se", "concavity_se", "symmetry_se", "fractal_dimension_se"]
# # num_feat = ["radius_worst", "perimeter_worst", "area_worst", "smoothness_worst", "concavity_worst", "symmetry_worst", "fractal_dimension_worst"]
# # num_feat = ["radius_mean", "perimeter_mean", "area_mean", "smoothness_mean", "concavity_mean", "symmetry_mean", "fractal_dimension_mean"]
# for col in num_feat:
#     print(f"Columns : {col}")
#     plt.hist(df[col], density=True, alpha=0.6, color='b')   # 绘制当前特征的直方图，用蓝色表示，透明度为0.6
#
#     xmin, xmax = plt.xlim()   # 获取当前直方图的x轴范围
#     x = np.linspace(xmin, xmax, 100)   # 在xmin和xmax之间生成100个线性间隔的点
#     p = stats.norm.pdf(x, np.mean(df[col]), np.std(df[col]))   # 计算这些点的正态分布概率密度函数值
#     plt.plot(x, p, 'k--', linewidth=2)   # 绘制正态分布曲线
#     plt.title(col)
#     plt.show()
#
#     stat, p = shapiro(df[col])   # 进行Shapiro-Wilk测试，检查数据的正态性
#     print("Statistics = %.3f, p = %.3f" % (stat, p))
#
#     alpha = 0.05   # 设置显著性水平，如果p值大于显著性水平，则数据符合高斯分布，否则不符合
#     if p > alpha:
#         print("数据符合高斯分布\n")
#     else:
#         print("数据不符合高斯分布\n")

import numpy as np
from scipy import stats
from scipy.stats import shapiro

num_feat = ["radius_se", "perimeter_se", "area_se", "smoothness_se", "concavity_se", "symmetry_se",
            "fractal_dimension_se"]
for col in num_feat:
    print(f"Columns : {col}")
    plt.hist(df[col], density=True, alpha=0.6, color='b')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(df[col]), np.std(df[col]))
    plt.plot(x, p, 'k--', linewidth=2)
    plt.show()

    stat, p = shapiro(df[col])
    print("Statistics = %.3f, p = %.3f" % (stat, p))

    alpha = 0.05
    if p > alpha:
        print("Dats looks Gaussian Distribution (fail to reject H0) \n")
    else:
        print("Data does not look Gaussian Distribution (reject H0) \n")

"""
Shapiro-Wilk测试是一种统计测试，用于检验一组数据是否服从正态分布。这里是关于Shapiro-Wilk测试的一些关键点：
原假设（H0）：数据集来自一个正态分布的总体。备择假设（H1）：数据集不来自一个正态分布的总体
测试通过比较样本数据与正态分布的期望值来进行。具体来说，它计算了一个名为W统计量的值，该值是基于样本数据的排序顺序和回归分析的概念。如果W统计量接近1，表明样本数据接近正态分布

在进行Shapiro-Wilk测试时，会计算出一个p值。这个p值用来决定是否拒绝原假设：
如果p值大于0.05，通常意味着没有足够的证据拒绝原假设，因此可以认为数据服从正态分布。如果p值小于0.05，则有足够的证据拒绝原假设，认为数据不服从正态分布
需要注意的是，Shapiro-Wilk测试对于小样本尤其敏感，适用于样本量较小的情况。对于大样本，可能会使用其他的正态性检验方法，如Kolmogorov-Smirnov检验
"""
# ### They need to be removed by IQR method.

# In[ ]:


# IQR method
for col in num_feat:
    Q1 = df[col].quantile(0.25)   # 计算第一四分位数
    Q3 = df[col].quantile(0.75)   # 计算第三四分位数
    IQR = Q3 - Q1   # 计算四分位数间距（IQR）
    df = df[(df[col] >= (Q1 - 1.5*IQR)) & (df[col] <= (Q3 + 1.5*IQR))]   # 更新数据框，只保留没有异常值的行
    # 使用1.5作为异常值阈值是基于统计学中的一个常用规则。这个规则来源于箱形图（boxplot），它是由著名统计学家约翰·图基（John Tukey）提出的
    # 在箱形图中，异常值通常被定义为低于Q1-1.5*IQR 或高于Q3+1.5*IQR的观测值
    # 如选用比1.5大的数，则只有更极端的值才会被识别为异常值；如果选用比1.5小的数，那么就会有更多的正常值被错误地认为是异常值
print(df)

"""
IQR方法，即四分位距（Interquartile Range），是一种统计学中用来衡量数据变异性和识别异常值的方法。它通过计算数据集的第一四分位数（Q1）和第三四分位数（Q3）之间的差值来实现
IQR是一种非常有用的统计量，因为它不受极端值的影响，这使得它成为一种稳健的异常值检测方法。在数据分析和数据清洗过程中，IQR方法可以帮助我们识别和处理那些可能扭曲分析结果的异常数据点

具体来说，IQR方法包括以下几个步骤：
将数据集从小到大排序
找到数据集的中位数，即第二四分位数（Q2），将数据集分为两部分
在数据集的下半部分中找到中位数，即为第一四分位数（Q1）
在数据集的上半部分中找到中位数，即为第三四分位数（Q3）
计算IQR值，即IQR=Q3-Q1
任何低于Q1-1.5IQR或高于Q3-1.5*IQR的数据都被视为异常
"""

"""
处理异常值的方法：
标准差法（Standard Deviation Method）: 使用数据的平均值和标准差来定义异常值。通常，超出平均值加减两个或三个标准差的数据点被认为是异常值
Z-分数法（Z-Score Method）: Z-分数表示数据点与平均值的距离，以标准差为单位。Z-分数绝对值高于2或3的数据点通常被视为异常值
DBSCAN聚类法（Density-Based Spatial Clustering of Applications with Noise）: 这是一种基于密度的聚类算法，可以识别任何形状的聚类，并将低密度区域中的点视为异常值
孤立森林法（Isolation Forest）: 这是一种基于树的算法，适用于高维数据集。它通过随机选择特征然后随机选择切分值来“孤立”观测值，异常值通常更容易被孤立
箱线图法（Boxplot Method）: 箱线图是一种用于显示数据分布的图形表示方法，它可以直观地显示出异常值
局部异常因子法（Local Outlier Factor, LOF）: 这种方法通过比较给定数据点与其邻居的局部密度偏差来识别异常值
"""

# In[ ]:


plt.figure(figsize=(30, 25))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(6, 5, i)
    skewness = df[col].skew()   # 计算当前列的偏度
    sns.distplot(df[col], kde=True, label="Skew = %.3f" %(skewness))   # 绘制当前特征的分布图，并在图例中显示偏度值
    plt.title(f"Skewness of {col} Data [outliers removed")
    plt.legend(loc="best")   # 显示图例，位置自动选择最佳位置
    plt.tight_layout()   # 调整子图布局，使之填满整个图形空间
    plt.plot()
plt.show()

# ### You see skewness that many outliers are removed by IQR method.

# In[ ]:


# Split dataset with train/test
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# 用train_test_split()函数将数据分割为训练集和测试集。test_size=0.25表示测试集占总数据的25%，random_state=0用于确保每次分割都能得到相同的结果

# # Machine Learning - Classification of diagnosis

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[ ]:


def accuracy(X_train, X_test, y_train, y_test):
    # 使用逻辑回归模型
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_reg = reg.predict(X_test)

    # 使用支持向量机模型
    svc = SVC()
    svc.fit(X_train, y_train)
    y_svc = svc.predict(X_test)

    # 使用决策树分类器
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_dtc = dtc.predict(X_test)

    # 使用随机森林分类器
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_rfc = rfc.predict(X_test)

    # 使用梯度提升分类器
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_gbc = gbc.predict(X_test)

    # 使用AdaBoost分类器
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_abc = abc.predict(X_test)

    # 使用K最近邻分类器
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_knc = knc.predict(X_test)

    # 计算每个模型的准确率并返回
    return accuracy_score(y_test, y_reg), accuracy_score(y_test, y_svc), accuracy_score(y_test, y_dtc), accuracy_score(y_test, y_rfc), accuracy_score(y_test, y_gbc), accuracy_score(y_test, y_abc), accuracy_score(y_test, y_knc)


"""
逻辑回归 (Logistic Regression)：
优点：简单、易于实现、计算效率高
缺点：在处理非线性问题时表现不佳，对于复杂的关系可能需要转换特征

支持向量机 (SVC)：
优点：在高维空间表现良好，适用于非线性问题
缺点：对大规模数据训练速度较慢，对参数和核函数选择敏感

决策树 (Decision Tree Classifier)：
优点：易于理解和解释，不需要数据预处理
缺点：容易过拟合，对于复杂的树可能需要剪枝

随机森林 (Random Forest Classifier)：
优点：减少了决策树的过拟合，提高了准确率
缺点：模型较大，需要更多的计算资源

梯度提升 (Gradient Boosting Classifier)：
优点：通常性能很好，可以处理各种类型的数据
缺点：训练时间较长，可能会过拟合

AdaBoost (AdaBoost Classifier)：
优点：提高了弱分类器的性能，易于实现
缺点：对异常值敏感，过拟合的风险

K最近邻 (KNeighbors Classifier)：
优点：简单有效，不需要假设数据分布
缺点：对大数据集计算量大，需要合适的距离度量
"""

# In[ ]:


#
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

    # 计算每个模型预测结果的精确度分数，并返回这些分数
    return precision_score(y_test, y_reg), precision_score(y_test, y_svc), precision_score(y_test, y_dtc), precision_score(y_test, y_rfc), precision_score(y_test, y_gbc), precision_score(y_test, y_abc), precision_score(y_test, y_knc)


"""
精确度（Precision）:精确度是指模型预测为正类（positive class）的样本中，实际上也为正类的比例，TP/(TP+FP)
准确度（Accuracy）:准确度是指模型正确预测的样本（无论正类或负类）占总样本的比例，(TP+TN)/(TP+FP+TN+FN)
其中，TN是真负例（true negatives）的数量，FN 是假负例（false negatives）的数量，TP是真正例（true positives）的数量，FP是假正例（false positives）的数量
"""

# In[ ]:


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

    # 计算每个模型预测结果的召回率
    return recall_score(y_test, y_reg), recall_score(y_test, y_svc), recall_score(y_test, y_dtc), recall_score(y_test, y_rfc), recall_score(y_test, y_gbc), recall_score(y_test, y_abc), recall_score(y_test, y_knc)

"""
召回率（Recall）衡量的是模型正确识别正类（positive class）的能力。具体来说，召回率是模型正确预测为正类的样本占所有实际正类样本的比例
召回率高意味着模型错过的正类样本较少，但这可能会伴随着更多的假正例（即，将负类错误地预测为正类）
召回率和精确度通常需要权衡，因为提高一个往往会降低另一个。为了综合考虑精确度和召回率，有时会使用F1分数，它是精确度和召回率的调和平均值
Recall=TP/(TP+FN)。TP是真正例（true positives）的数量，FN是假负例（false negatives）的数量
"""

# In[ ]:


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

    # 计算每个模型的F1分数
    return f1_score(y_test, y_reg), f1_score(y_test, y_svc), f1_score(y_test, y_dtc), f1_score(y_test, y_rfc), f1_score(y_test, y_gbc), f1_score(y_test, y_abc), f1_score(y_test, y_knc)


"""
F1分数是用于衡量分类模型性能的指标，特别是在数据类别不平衡的情况下。它是精确度（Precision）和召回率（Recall）的调和平均值
F1分数的取值范围是0到1，1表示完美的精确度和召回率，0表示两者中至少有一个为零。F1分数越高，模型的性能越好
这个指标对于那些错误分类的代价很高的应用场景特别有用，比如医疗诊断或欺诈检测
F1分数的计算公式是F1 = 2*Precision*Recall / (Precision+Recall)
"""

# In[ ]:


scores = pd.DataFrame(columns=["REG", "SVC", "DTC", "RFC", "GBC", "ABC", "KNC"],
                     index=["ACC", "PREC", "REC", "F1"])
acc = [reg, svc, dtc, rfc, gbc, abc, knc] = accuracy(X_train.values, X_test.values, y_train.values, y_test.values)
scores.iloc[0, :] = acc
prec = [reg, svc, dtc, rfc, gbc, abc, knc] = precision(X_train.values, X_test.values, y_train.values, y_test.values)
scores.iloc[1, :] = prec
rec = [reg, svc, dtc, rfc, gbc, abc, knc] = recall(X_train.values, X_test.values, y_train.values, y_test.values)
scores.iloc[2, :] = rec
f_1 = [reg, svc, dtc, rfc, gbc, abc, knc] = f1(X_train.values, X_test.values, y_train.values, y_test.values)
scores.iloc[3, :] = f_1
print(scores)


# ### RandomForestClassifier has highest score in 7 models.
# ### Let's find Train/Test model score of RandomForestClassifier.

# In[ ]:


train_scores = []
test_scores = []

for i in range(2, 10):
    # 设置随机森林分类器，树的数量为50*i，最大特征数为i/10，树的最大深度为i，分裂内部节点所需的最小样本数为i
    rfc = RandomForestClassifier(n_estimators=i * 50, max_features=i/10, max_depth=i, min_samples_split=i)
    rfc.fit(X_train, y_train)   # 训练分类器
    train_scores.append(rfc.score(X_train, y_train))   # 计算并储存训练集得分
    test_scores.append(rfc.score(X_test, y_test))   # 计算并储存测试集得分

# 绘制训练得分的折线图，用*和蓝色线条标记训练集得分，用o和红色线条标记测试集得分
sns.lineplot(train_scores, marker='*', color='b')
sns.lineplot(test_scores, marker='o', color='r')


"""
随机森林分类器是一种集成学习方法，它通过结合多个决策树的预测结果来提高整体模型的准确性和稳定性

随机森林分类器的一些关键特点：
集成学习：随机森林属于集成学习算法，它通过构建多个决策树并将它们的预测结果进行汇总来做出最终决策
决策树：决策树是一种基本的分类与回归方法，它通过学习样本特征与输出标签之间的关系来构建一个树形结构
Bagging：随机森林使用Bagging（自助聚合）技术来训练每一棵树，即从原始数据集中通过有放回抽样选取多个子样本集来训练每一棵树
特征随机性：在构建每棵树的过程中，随机森林算法会在每个分裂节点随机选择一部分特征，这增加了模型的多样性，减少了过拟合的风险
多数投票：在分类任务中，随机森林通过多数投票的方式来确定最终的类别，即每棵树给出一个预测结果，最终结果是得票最多的类别

随机森林的优点包括：
高准确率：由于集成了多个决策树，随机森林通常能够提供较高的准确率
防止过拟合：通过特征随机性和Bagging技术，随机森林能够有效防止模型过拟合
适用性广：随机森林可以处理分类和回归任务，适用于各种类型的数据集

随机森林的缺点包括：
模型解释性差：由于模型是由多棵树组成的，因此难以解释模型是如何做出具体决策的
计算量大：训练多棵树需要较大的计算资源，尤其是在处理大型数据集时
"""

# In[ ]:


# i = 2
rfc = RandomForestClassifier(n_estimators=100, max_features=0.2, max_depth=2, min_samples_split=2)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("随机森林训练集得分 :", rfc.score(X_train, y_train))
print("随机森林测试集得分:", rfc.score(X_test, y_test))
print("----------")
print("随机森林准确度:", accuracy_score(y_test, y_pred))
print("随机森林精确度:", precision_score(y_test, y_pred))
print("随机森林召回率:", recall_score(y_test, y_pred))
print("随机森林F1得分:", f1_score(y_test, y_pred))


# In[ ]:
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
n_trees = len(rfc.estimators_)
tree_index = 0
single_tree = rfc.estimators_[tree_index]

print(single_tree)

dot_data = export_graphviz(single_tree, out_file=None, feature_names=X.columns,
                           class_names=['M', 'B'], filled=True, rounded=True, special_characters=True)
print(dot_data)
