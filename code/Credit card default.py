#!/usr/bin/env python
# coding: utf-8

# In[109]:


from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">HERE</a>.''')


# ### Mathematics in Machine Learning
# ### Sofia Perosin, S269748

# # Credit Card Default

# ## Table of contents
# 1. [Problem definition](#problemdefinition)
# 2. [Provided data](#provideddata)
#     1. [Attributes](#attributes)
# 3. [Analyze Data](#analyzedata)
#     1. [Data Structure](#datastructure) 
#     2. [Data Distribution](#datadistribution)
#     3. [Data Visualization](#datavisualization) 
#     4. [Data Exploration](#dataexploration)
#         1. [Pairwaise scatterplots](#pairwaise) 
#         2. [Correlation matrix](#corrmatrix)
#         3. [Outlier](#outliers)
#             1. [Isolation Forest](#isolationforest)
# 4. [Prepare Data](#preparedata)
#     1. [Preprocess Data](#preprocess) 
#     2. [Transform Data](#transform)
#         1. [Scaling](#scaling) 
#         2. [PCA](#pca)
# 5. [Evaluate Algorithms](#evaluation)
#     1. [Cross Validation](#crossvalidation)
#     2. [Undersampling](#undersampling)
#         1. [Random Forest](#randomforest)
#         2. [Logistic Regression](#logisticregression)
#         3. [Support Vector Machine](#svm)
#             1. [Linear SVM](#linearsvm)
#             2. [Rbf SVM](#rbfsvm)
#             3. [Poly SVM](#polysvm)
#         4. [Gaussian NB](#gaussian)
#         5. [Linear Discriminant Analysis](#lda)
#     3. [Oversampling SMOTE](#oversampling)
#         1. [Random Forest](#randomforestover)
#         2. [Logistic Regression](#logisticregressionover)
#         3. [Support Vector Machine](#svmover)
#             1. [Linear SVM](#linearsvmover)
#             2. [Rbf SVM](#rbfsvmover)
#             3. [Poly SVM](#polysvmover)
#         4. [Gaussian NB](#gaussianover)
#         5. [Linear Discriminant Analysis](#ldaover)
# 6. [Conclusion](#conclusion)
# 7. [References](#references)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss

import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# In[2]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# ## Problem definiton <a name="problemdefinition"></a>

# <p>The goal of this project is to try different algorithms in order to understand how accurate they are in detecting whether a client will do a credit card default.</p>
# <p>Banks have to carefully check customers in order to avoid financial losses.<br>
# To do that they need to be able to understand if a customer will present a default credit card status. <br>
# So for this task a good classifier is the one which is able to correctly classify a default payment, since it represents the financial loss of the bank.<br>
# Obviously also the <i>non default</i> clients have to be correctly classified, otherwise the bank will lose them as clients.</p>

# ## Provided data <a name="provideddata"></a>

# <p>The dataset used to complete this task is <i>default of credit card clients</i> from UCI machine learning repository.<br>
# It contains 30000 instances, each one describes a client of a Taiwan bank, taking into account different attributes (24 precisely).</p>

# ### Attributes  <a name="attributes"></a>
# <ul>
#     <li> <b>LIMIT BAL</b>: It's the amount of the given credit, in NT dollar. </li>
#     <li> <b>SEX</b>: 1=male, 2=female.</li>
#     <li> <b>EDUCATION</b>: 1=graduate school, 2=university, 3=high school, 4=others.</li>
#     <li> <b>MARRIAGE</b>: 1=married, 2=single, 3=others.</li>
#     <li> <b>AGE</b>: age of client.</li>
#     <li> <b>PAY_0, ...., PAY 6</b>:history of past payments:</li>
#         <ul>
#             <li> PAY_0 repayment status in September 2005, .... PAY_6 repayment status in April 2005 </li>
#             <li>-1=pay duly</li>
#             <li>1=payment delay for one month</li>
#             <li>2=payment delay for two months</li>
#                 .....
#             <li>9=payment delay for nine months or more</li>
#         </ul>
#     <li><b>BILL_AMT1, ...., BILL_AMT6</b>: amount of bill statement, in NT dollar:</li>
#         <ul>
#             <li>BILL_AMT1 it's in September 2005, .... BILL_AMT6 it's in April 2005</li>
#         </ul>
#     <li><b>PAY_AMT1, ...., PAY_AMT6</b>: amount of previous payment, in NT dollar:</li>
#     <ul>
#         <li>PAY_AMT1 it's in September 2005, .... PAY_AMT6 it's in April 2005</li>
#     </ul>
#     <li><b>DEFAULT PAYMENT NEXT MONTH</b>: 0=No, 1=Yes</li>
# </ul>

# ## Analyze Data <a name="analyzedata"></a>

# ### Data Structure <a name="datastructure"></a>

# In[3]:


df=pd.read_excel("credit_card.xls",index_col=0)
column_name=df.iloc[0]
credit_card = df[1:].copy()
credit_card.columns = column_name
credit_card=credit_card.astype("int64")


# In[4]:


lista=credit_card.isnull().sum(axis = 0)
missing=False
for el in lista:
    if el!=0:
        missing=True
if missing:
    print("ATTENTION: there are missing values!")
else:
    print("There are not missing values.")


# In[5]:


print(f"Value in SEX attribute column: {set(credit_card['SEX'])}")
print(f"Value in EDUCATION attribute column: {set(credit_card['EDUCATION'])}")
print(f"Value in MARRIAGE attribute column: {set(credit_card['MARRIAGE'])}")
print(f"Value in RESPONSE (YES or NO) column: {set(credit_card['default payment next month'])}")
print(f"Value in PAY_0 column: {set(credit_card['PAY_0'])}")      
print(f"Value in PAY_2 column: {set(credit_card['PAY_2'])}")      
print(f"Value in PAY_3 column: {set(credit_card['PAY_3'])}")      
print(f"Value in PAY_4 column: {set(credit_card['PAY_4'])}")      
print(f"Value in PAY_5 column: {set(credit_card['PAY_5'])}")      
print(f"Value in PAY_6 column: {set(credit_card['PAY_6'])}")      


# <p>So looking at the values present in the attributes printed before some changes have to be done:<br>
# <ul>
#     <li>Attribute <i>marriage</i> should present only one of those values: 1,2,3; but in the dataset some records have value 0.</li>
#     <li>Attribute <i>education</i> should present only on of those values: 1,2,3,4; but in the dataset some records have values 0,5,6.</li>
# </ul>       
# Since in both cases there is an attribute which represent the class <i>Other</i> (respectively 3 for <i>marriage</i> and 4 for <i>education</i>) all the attributess not-known are mapped in that category.
# <ul>
#     <li>Attributes <i>PAY_N</i> should present only on of those values: -1,1,2,3,4,5,6,7,8,9; but in the dataset some records have value -2 and 0.</li></ul>
# In this case, in order to use this attributes as a numerical attribute, and not a categorical one, all the values -2 and -1 are mapped in 0. In this way <i>PAY_N</i> will indicate for how many months the payment was delayed.

# In[6]:


credit_card["MARRIAGE"]=credit_card["MARRIAGE"].replace(0,3)
credit_card["EDUCATION"]=credit_card["EDUCATION"].replace([0,5,6],4)
credit_card["PAY_0"]=credit_card["PAY_0"].replace([-2,-1],0)
credit_card["PAY_2"]=credit_card["PAY_2"].replace([-2,-1],0)
credit_card["PAY_3"]=credit_card["PAY_3"].replace([-2,-1],0)
credit_card["PAY_4"]=credit_card["PAY_4"].replace([-2,-1],0)
credit_card["PAY_5"]=credit_card["PAY_5"].replace([-2,-1],0)
credit_card["PAY_6"]=credit_card["PAY_6"].replace([-2,-1],0)


# ### Data Distribution <a name="datadistribution"></a>

# In[7]:


credit_card.describe()


# <p>It's possible to see how for example:
#     <ul>
#         <li>there is a huge difference on the amount of the given credit to each client (look at the std of <i>LIMIT_BAL</i> attribute).</li>
#         <li>there are more women than men (since the mean of attribute <i>SEX</i> is 1.6) but for this kind of information more understandable graphs will be shown later.</li></ul></p>

# ### Data Visualization <a name="datavisualization"></a>

# #### Attribute Histograms

# In[8]:


x=np.arange(2)
yes=credit_card[credit_card["default payment next month"]==1].shape[0]
no=credit_card[credit_card["default payment next month"]==0].shape[0]
y=[yes,no]
fig,ax=plt.subplots(figsize=(15,6))
ax.set_title("DEFAULT PAYMENT",fontsize=15)
ax.set_ylabel("Number of customers",fontsize=10)
plt.xticks(x,("YES","NO"),fontsize=10)
plt.bar(x,y,color=["red","green"])
tot=yes+no
for i in range(len(x)):
    ax.annotate((y[i],("%.2f "%((y[i])/tot*100))+"%"),(x[i],y[i]),xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=11)
plt.show()


# How it's possible to see the two classes (default and no default) are highly unbalanced.<br>
# From this some considerations could be done:
# <ul>
#     <li>Since classes are unbalanced good metrics to use to evaluate the model are: <i>precision</i>, <i>recall</i> and <i>fscore</i>.<br>
#         A metric like <i>accuracy</i> in this case is meaningless.</li>
#     <li>To overcome this problem different strategies can be implemented:
#         <ul>
#             <li>undersampling: This method consists into undersampling the class with the higher cardinality.</li>
#             <li>oversampling. This method instead consists into oversampling the smaller class</li></ul>
#         Further explanation will be given when these methods will be used.</li>
# </ul>
# 
# Since the dataset is not so small, by adopting an undersampling strategy should not be lost a lot of information.<br>
# At the same time by oversampling the smaller class this allows to not loose any information, but in this case synthetic data are used, so this should be taken into account.<br>
# It's not so easy to understand what should be done, in most of the cases the best practice is the one to ask to a domain expert what we should do.<br>

# In[9]:


def plot0(colonna):
    yes=credit_card[credit_card["default payment next month"]==1]
    no=credit_card[credit_card["default payment next month"]==0]
    
    distinct_amount_YES=set(yes[colonna])
    lista_amount_YES=sorted(distinct_amount_YES)
    count_per_limit_YES=[]
    conti_YES=yes[colonna].value_counts()
    for amount in distinct_amount_YES:
        count_per_limit_YES.append(conti_YES[amount])

    distinct_amount_NO=set(no[colonna])
    lista_amount_NO=sorted(distinct_amount_NO)
    count_per_limit_NO=[]
    conti_NO=no[colonna].value_counts()
    for amount in distinct_amount_NO:
        count_per_limit_NO.append(conti_NO[amount])


    fig, ax = plt.subplots(figsize=(11.5,4))
    ax.scatter(lista_amount_YES, count_per_limit_YES, c="red", label="default payment = YES")
    ax.scatter(lista_amount_NO,count_per_limit_NO, c="green", label="default payment = NO")
    ax.set_title(colonna,fontsize=12)
    ax.set_ylabel("Number of customers",fontsize=10)
    ax.legend(fontsize=12)
    plt.show()


# Distributions of <i>LIMIT BAL</i>, <i>AGE</i>, <i>SEX</i>, <i>EDUCATION</i> and <i>MARRIAGE</i> attributes according to the value of <i>default payment</i>:

# In[10]:


plot0("LIMIT_BAL")
plot0("AGE")


# In[11]:


def plot2(colonna,nomi):
    possible_values=set(credit_card[colonna])
    x=np.arange(len(possible_values))
    
    y_tot=[]
    for value in possible_values:
        count=credit_card[credit_card[colonna]==value]
        y_tot.append(count)
    
    y_yes=[]
    y_no=[]
    for category in y_tot:
        yes=category[category["default payment next month"]==1].shape[0]
        no=category[category["default payment next month"]==0].shape[0]
        y_yes.append(yes)
        y_no.append(no)
    
    width = 0.35

    fig,ax=plt.subplots(figsize=(10,4))
    rects1 = ax.bar(x - width/2, y_yes, width, label="default payment = YES",color="red")
    rects2 = ax.bar(x + width/2, y_no, width, label="default payment = NO",color="green")

    ax.set_title(colonna,fontsize=12)
    ax.set_ylabel("Number of customers",fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(nomi,fontsize=10)
    ax.legend()


    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),textcoords="offset points",ha='center', va='bottom',fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()

    plt.show()


# In[12]:


plot2("SEX",["male","female"])
plot2("EDUCATION",["graduate school","university","high school","others"])
plot2("MARRIAGE",["married","single","others"])


# General distribution of <i>SEX</i>, <i>EDUCATION</i> and <i>MARRIAGE</i> attributes:

# In[13]:


def plot1(colonna,nomi):
    x=np.arange(len(set(credit_card[colonna])))
    possible_values=set(credit_card[colonna])
    y=[]
    for value in possible_values:
        count=credit_card[credit_card[colonna]==value].shape[0]
        y.append(count)
    fig,ax=plt.subplots(figsize=(11.5,4))
    ax.set_title(colonna,fontsize=12)
    ax.set_ylabel("Number of customers",fontsize=10)
    if colonna=="SEX":
        colori=["cyan","pink"]
    else:
        colori=["yellow","blue","magenta","orange"]
        colori=colori[:len(nomi)]
    plt.bar(x,y,color=colori)
    plt.xticks(x,(nomi),fontsize=10)
    for i in range(len(x)):
        ax.annotate(y[i],(x[i],y[i]),xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=8)
    plt.show()


# In[14]:


plot1("SEX",["male","female"])
plot1("EDUCATION",["graduate school","university","high school","others"])
plot1("MARRIAGE",["married","single","others"])


# ### Data Exploration <a name="dataexploration"></a>

# #### Pairwaise scatterplots of attributes <a name="pairwaise"></a>
# This representation is very useful to understand the relationship between the dimensions of the dataset when they are more than 3 (since scatter plot works only in 2D and 3D cases).

# In[15]:


plot=sn.pairplot(credit_card,vars=["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE"],hue="default payment next month",palette=["green","red"])


# In[16]:


credit_card_dummies=credit_card
credit_card_dummies=pd.concat([credit_card,pd.get_dummies(credit_card["SEX"],prefix="SEX")],axis=1)
credit_card_dummies=credit_card_dummies.drop(["SEX"],axis=1)
credit_card_dummies=pd.concat([credit_card_dummies,pd.get_dummies(credit_card["EDUCATION"],prefix="EDUCATION")],axis=1)
credit_card_dummies=credit_card_dummies.drop(["EDUCATION"],axis=1)
credit_card_dummies=pd.concat([credit_card_dummies,pd.get_dummies(credit_card["MARRIAGE"],prefix="MARRIAGE")],axis=1)
credit_card_dummies=credit_card_dummies.drop(["MARRIAGE"],axis=1)


# #### Correlation matrix of attributes <a name="corrmatrix"></a>
# <p>Correlation matrix, as pairwise scatterplots, allows to understand the relationships among attributes; and thanks to correlation coefficients a precise measure of these relations is given.</p>
# <p>Correlation coefficients are calculated throught Pearson's method, and so according to the following formula:<br>
# $\rho_{X,Y}=\frac{cov(X,Y)}{\sigma_{X}\sigma_{Y}} $ <br>
#     where $cov(X,Y)$ is the covariance between <i>X</i> and <i>Y</i>, while $\sigma_{X}$ is the standard deviation of <i>X</i> and $\sigma_{Y}$ is the standard deviation of <i>Y</i>.<br>
# For the sake of completeness:
# <ul>
#     <li>$cov(X,Y)=E [ (X-E[X])(Y-E[Y]) ]$</li>
#     <li>$\sigma_{X}=\sqrt{E[(X-\mu)^{2}]}=\sqrt{E[X^{2}]-(E[X])^{2}}$</li>
#     </ul>
# </p>
# <p>The interpretation of the obtained coefficient is the following:
#     <ul>
#         <li>$\rho=1$ implies that the relation between <i>X</i> and <i>Y</i> is described by a linear equation in a perfect way, and all data points lie on a line for which <i>Y</i> increases as <i>X</i> increases.</li>
#         <li>$\rho=-1$ implies that the relation between <i>X</i> and <i>Y</i> is described by a linear equation in a perfect way, and all data points lie on a line for which <i>Y</i> decreases as <i>X</i> increases.</li>
#         <li>$\rho=0$ implies that there is no linear correlation between <i>X</i> and <i>Y.</i></li>
#         </ul>
# </p>
# <p>Note that to properly understand the correlations among attributes, categorical attributes like <i>marriage</i>, <i>sex</i> and <i>education</i> must be converted into dummy variables.</p>

# In[17]:


fig,ax=plt.subplots(figsize=(30,30))
corrMatrix = credit_card_dummies.corr()
sn.heatmap(corrMatrix,annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# #### Outlier <a name="outliers"></a>
# <p>The last step of data exploration is looking for outliers.</p>
# <p>This step is very important because they could affects the performances of the model (in particular of those models which use equations).<br>
# Sometimes outliers can be legitimate extreme values, but could happens that they are the results of bad data collection or misdialed values (for example a negative price).<br>
# In the latter case they have to be removed.</p>
# <p>A very useful tool to detect possible outliers is the box plot:<br>
#     box plot is a way of displaying the dataset based on a five-number summary:
#     <ul>
#         <li><b>Minimum</b>: it's the lowest data point excluding any outliers.</li>
#         <li><b>First quartile</b>: it's the lower quartile, and it's the median of the lower half of the dataset.</li>
#         <li><b>Median</b>: it's the middle value of the dataset.</li>
#         <li><b>Third quartile</b>: it's the upper quartile, and it's the median of the upper half of the dataset.</li>
#         <li><b>Maximum</b>: it's the largest data point excluding any outliers.</li>
#     </ul>
# All the data over the whiskers could be outliers.
# </p>
# <img src="img/boxplot.png">

# Here are analyze the boxplots of numerical features; at this step any standardization is applied in order to understand if a possible outlier has a reasonable value or not:

# In[18]:


study_possible_outlier=credit_card[['LIMIT_BAL','AGE', 'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].copy()


# In[19]:


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('LIMIT_BAL Box plot')
data=[study_possible_outlier['LIMIT_BAL']]
ax.boxplot(data)
plt.show()


# Looking at this box plot there doesn't seem to be outliers:<br>
# <ul>
#     <li>All the values are grater or equal than zero.
#     </li>
#     <li>The biggest value is 1000000 NT which corresponds to about 30000 euros, a value not so high; so it will be not considered as an outlier.</li>
#     </ul>
#     

# In[20]:


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('AGE Box plot')
data=[study_possible_outlier['AGE']]
ax.boxplot(data)
plt.show()


# Also in this case there doesn't seem to be outliers:<br>
# <ul>
#     <li>All the customers are more than 20 years old.
#     </li>
#     <li>The biggest value is 80 year, which is a reasonable value.</li>
#     </ul>
#     

# In[21]:


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('BILL AMT Box plot')
data=[study_possible_outlier['BILL_AMT1'],study_possible_outlier['BILL_AMT2'],study_possible_outlier['BILL_AMT3'],study_possible_outlier['BILL_AMT4'],study_possible_outlier['BILL_AMT5'],study_possible_outlier['BILL_AMT6'],]
ax.boxplot(data)
plt.show()


# This attribute represents the bill statement and it's not so easy to understand if these are outliers in the sense of wrong data collection or they are simply extreme values:
# <ul>
#     <li>Negative values indicate credit card debt or maybe a bank loan.</li>
#     <li>Very high positive values indicate that the client is rich and he can use that amount of money.</li>
# </ul>
# So in my opinion these values should be considered as right values.

# In[22]:


fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('PAY AMT Box plot')
data=[study_possible_outlier['PAY_AMT1'],study_possible_outlier['PAY_AMT2'],study_possible_outlier['PAY_AMT3'],study_possible_outlier['PAY_AMT4'],study_possible_outlier['PAY_AMT5'],study_possible_outlier['PAY_AMT6']]
ax.boxplot(data)
plt.show()


# This attribute represents the amount of the previous payments:
# <ul>
#     <li>All the values are grater or equal zero, and this is good since a negative payment would not make sense .</li>
#     <li>Very high positive values indicate that the client has done important expenses, but also 1750000 NT (about 52580 euros) could be a real value.</li>
# </ul>
# So in my opinion these values should be considered as right values.

# <p>It can be also very useful to visualize the boxplot of all the numerical features together; remembering to normalize them otherwise the representation is meaningless.</p>

# In[23]:


study_possible_outlier_transformed = Normalizer().fit_transform(study_possible_outlier)
study_possible_outlier_transformed_df=pd.DataFrame(study_possible_outlier_transformed,columns=study_possible_outlier.columns)


# In[24]:


fig, ax = plt.subplots(figsize=(15,10))
ax.grid()
ax = sn.boxplot(data=study_possible_outlier_transformed_df)


# <p>It's possible to see that the biggest variation is visible among <i>PAY_AMTi</i> features, instead the others are more stable.

# #### Isolation Forest <a name="isolationforest"></a>
# <p>The method previously seen is useful to detect outliers in a univariate way: looking at one feature at time, but it could be useful to try to detect possible outliers by considering all the features at the same time.</p>
# <p>Isolation forest allows to do it: the main idea is that outliers are few and far from others observation.<br></p><br>
# <b>How it works:</b>
# <ul> 
#     <li>Pick randomly a feature and randomly splits according to a value between its minimum and maximum.</li>
#     <li>The process is done recursively until a node has a single element.</li>
#     <li>The path length from the root node to the terminating node is equivalent to the number of splitting required to isolate the observation.</li>
#     <li>The path length is the measure of normality and it's the decision function that will classify each observation as outliers or not (the idea is that if an observation is isolated, this means that it's easy to separate and so it could be an outlier).</li>
#     <li>The estimation of the anomaly score for a given instance x is given by:<br><br>
#     $S(x,n)=2^{-\frac{E(h(x))}{C(n)}}$</li><br>
#     <ul><li>$h(x)$ is the path length, so the number of splits needed to isolate x.</li>
#     <li>$E(h(x))$ is the expected value of $h(x)$.</li>
#     <li>$C(n)$ is the maximum path length from the root to a leaf.</li>
#     <li>$n$ is the number of leaves.</li></ul>
#     <li>If $S(x,n)$ is close to 1, then x is very likely to be an anomaly</li>
#     <li>If $S(x,n)$ is smaller than 0.5, then x is likely to be a normal value</li>
# <br>
# In Sklearn implementation, the scores returned work in a different way: the lower the score, the more abnormal is the observation.<br><br>
# Because of the unbalancing of the classes, are removed only the "outliers" which belongs to <i>NO DEFAULT</i> class.

# In[25]:


df_isolation_data=credit_card_dummies[['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6','SEX_1', 'SEX_2', 'EDUCATION_1', 'EDUCATION_2', 'EDUCATION_3','EDUCATION_4', 'MARRIAGE_1', 'MARRIAGE_2', 'MARRIAGE_3']].copy()
isolationForestDetector = IsolationForest(random_state=0).fit(df_isolation_data)
scores=isolationForestDetector.predict(df_isolation_data)


# In[26]:


classiTot=credit_card_dummies["default payment next month"]
index_to_eliminate=[]
conto=0
for s in scores:
    if s==-1:
        classe=classiTot.iloc(0)[conto]
        if classe==0:
            index_to_eliminate.append(conto)
    conto=conto+1


# In[27]:


credit_card_dummies.drop(credit_card_dummies.index[index_to_eliminate],inplace=True)


# ## Prepare Data <a name="preparedata"></a>

# ### Preprocess Data <a name="preprocess"></a>
# <p>To properly evaluate the algorithm the entire dataset has to be divided into train and test set before applying any preprocessing step and before also undersampling or oversampling techniques, otherwise the result will be meaningless since it's evaluated on a different data distribution.</p>
# <p>To divide the original dataset in the training and testing part is used <i>"train_test_split"</i> method of the library <i>sklearn</i>, which is very useful because it allows to specify if data have to be split in a stratified fashion: so in this case data are splitting in order to mantain the same proportion of <i>YES</i> and <i>NO</i> default. 

# In[28]:


df_train,df_test=train_test_split(credit_card_dummies,train_size=0.80,shuffle=True,stratify=credit_card_dummies['default payment next month'],random_state=42)

label_train=df_train["default payment next month"].values.tolist()
df_train=df_train.drop(columns=["default payment next month"])

label_test=df_test["default payment next month"].values.tolist()
df_test=df_test.drop(columns=["default payment next month"])


# ### Transform Data <a name="transform"></a>
# 

# #### Scaling <a name="scaling"></a>
# <p>When data has a large number of features it is often useful to look for a lower-dimensional representation which preserves most of its properties, the most widely used techniques to do this are PCA and SVD. </p>
# <p>Thinking about PCA (explained better below): its aim is to find the components which maximize the variance, but if one feature varies less than another (because of the scale for example) PCA might determine that the direction of maximal variance more closely corresponds with the feature axis having the highest variance, even if it's not correct. <br>
# So <i>scaling</i> is an important step to do in order to be sure that a dimensionality reduction technique works in a proper way.</p>

# #### PCA <a name="pca"></a>
#  **P**rincipal **C**omponent **A**nalysis. <br>
# - The aim of PCA is to project the input data into a lower dimensional linear subspace which minimizes the *reconstruction error*. So it looks for finding the compression matrix *W* and the recovering matrix *U* in such a way that the total squared distance is minimal between the original and the recovered vecrors. <br>
# - Mathematically speaking the PCA aims to solve the following optimization problem: <br>
# $ \underset{W \in \mathbb{R}^{n,d}, U \in \mathbb{R}^{d,n} }{\arg\min}\sum_{i=1}^{m}\left \| x_{i} - UW{x}_{i}  \right \|^{2}_{2} $ <br>
# <br>
# - PCA can be also interpretated as a variance maximization operation: <br>
#     - An equivalent way, of the one shown before, of deriving the principal components is to find the projections which maximize the variance:
#         - The first principal component is the direction along which projections have the largest variance;
#         - The second principal component is the direction which maximize the variance among all the directions orthogonal to the first;
#         - and so on...
#     - In other words, since variance means information thanks to the directions of maximal variance it's possible to project the input data in a lower-dimensional space (represented by the principal components) where the most meaningful information are preserved; and so the reconstruction error is minimized.
#         - $ {1}^{st}$ principal component is the normalized ( $ \sum_{i=1}^{p}{\phi_{i1}}^{2}=1 $ ) linear combination: <br> $ Z_{1} = \phi_{11}X_{1} + \phi_{21}X_{2} + .... + \phi_{p1}X_{p} $ <br>
#         which has the largest variance
#             - $\phi_{1}$ is the *loading vector* with elements $\phi_{11}, \phi_{21}, .... , \phi_{p1} $ (defines a direction in the feature space along which the data vary the most)
#             - $z_{11}, z_{21}, .... , z_{m1} $ are the *scores* (the projections of the m points into the direction $\phi_{1}$)
#             - It has to be found the linear combination of the sample feature values of the form <br>
#             $ z_{i1} = \phi_{11}x_{i1} + \phi_{21}x_{i2} + .... + \phi_{p1}x_{ip} $ for *i* = 1, ..., *m* <br>
#             that has the largest sample variance. <br>
#             Since all $x_{ij}$ has mean zero, then also $z_{i1}$, and this means that the sample variance of $z_{i1}$ can be written as <br> 
#             $\frac{1}{m} \sum_{i=1}^{m} z_{i1}^2$ <br>
#             - So it corresponds to solve the optimization problem <br>
#             $ \underset{\phi_{11}, \phi_{21}, .... , \phi_{p1} }{maximize}\frac{1}{m} \sum_{i=1}^{m}(\sum_{j=1}^{p}{\phi_{j1}x_{ij}})^2$ subject to $ \sum_{j=1}^{p}{\phi_{j1}}^{2}=1$
#         - $ {2}^{nd}$ principal component: the reasoning is the same, the only difference is that it's a linear combination that has the maximal variance among all linear combinations that are **uncorrelated** with $Z_{1}$.
#         - and so on for the other components.

# <p>So now the PCA is applied only on numerical attributes, and not to the ones which in origin were categorical attributes.</p>
# <p>To understand which is a good number of components to use it has to be looked the graph of exaplained variance.<br>
# In this case the number of components choosen is the one which allows to reach an explained variance of 80%.</p>

# In[29]:


features = ['LIMIT_BAL', 'AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6']


scaler = StandardScaler()
scaler.fit(df_train[features])
X = scaler.transform(df_train[features])

pca = PCA(n_components = 20)
pca.fit(X)

cum_variance = np.cumsum(pca.explained_variance_ratio_)
idx1 = np.argmax(cum_variance > .80)
value=cum_variance[idx1]

x_plot=np.linspace(1,len(cum_variance),len(cum_variance))

fig,ax=plt.subplots(figsize=(15,6))
ax.set_title("Explained variance",fontsize=15)
ax.set_ylabel("Explained variance ratio",fontsize=10)
ax.set_xlabel("Number of components",fontsize=10)
plt.xticks(np.arange(1,21))
ax.plot(x_plot,cum_variance)
plt.scatter(idx1+1,value)
string=f"{idx1+1} components"
ax.annotate(string,(idx1,value),xytext=(-20,10),textcoords="offset points",ha="center",va="bottom",fontsize=15)
ax.grid(True)
plt.savefig("reviews_train.png")
plt.show()


# In[30]:


features = ['LIMIT_BAL', 'AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6']

scaler = StandardScaler()
scaler.fit(df_train[features])
X_train = scaler.transform(df_train[features])
X_test = scaler.transform(df_test[features])

pca = PCA(n_components = 9)
transformed_train=pca.fit_transform(X_train)
transformed_test=pca.fit_transform(X_test)


# In[31]:


for count in range(0,9):
    stringa="PCA_"+str(count+1)
    df_train[stringa]=transformed_train[:,count]
    df_test[stringa]=transformed_test[:,count]
    
    
df_train=df_train.drop(columns=['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
df_test=df_test.drop(columns=['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])


# ## Evaluate Algorithms <a name="evaluation"></a>

# ### Cross Validation <a name="crossvalidation"></a>
# <p>Cross validation is a widely used approach for estimating test error (which is the average error that results from using a statistical learning method to predict the target on a new observation, that was not present in the training set).</p>
# <p>When undersampling or oversampling techniques are used, cross validation has to be used carefully, because if those techniques are done before cross validation, all this will cause data leakage problem since the probability density function is altered.</p>
# <p>For this reason an ad-hoc process is built:
#     <ul>
#         <li><b>Stratified K-Fold</b> is used to identify the indices to split data in k equal-sized parts preserving the percentage of samples for each class.</li>
#         <li>One part k is left out.</li>
#         <li>On the other K-1 parts (combined) undersampling or oversampling techniques are applicated and the model is fit.</li>
#         <li>For the left k part are obtained the predictions.</li>
#         <li>This is done in turn for each part k=1,2,..,K and then the results are combined.</li>
#         </ul><p>
#     <br>
# <p>The score used to select the best configuration for each classifier is <b>f1_score</b>, according to the task.<br>
# <ul><li>Evaluating a model through accuracy could be useless: most of the records belong to class 0 (no default), if all the data are classified as 0, the accuracy will be high, but this will cause an huge financial loss for the bank, since any default is prevented.</li>
#     <li><b>F1 score</b> instead is a weighted average of the recall and precision:
#         <ul><br><li>$F1 score = 2 \times \frac{precision \times recal}{precision + recall}$ </li><br>
#             <li>$precision = \frac{True Positive}{True Positive + False Positive}$ </li><br>
#             <li>$recall = \frac{True Positive}{True Positive + False Negative}$ </li>
#         </ul><br>
#         And since precision is useful to minimize false positve and recall is useful to minimize false negative, f1 score is the best measure to use for this problem, in order to avoid as many false prediction as possible.
# </ul></p>
# 

# ### Undersampling <a name="undersampling" ></a>
# <p>The first technique used to deal with the unbalance dataset is undersampling, in particular <b>Random undersampling</b>.</p>
# <p>This method consists in remove samples from the majority class randomly.<br>
#     It must taken into account that it may discard useful samples.<br></p>
# <p>Pratically, from the points belonging to the majority class, a random sample is taken (without replacement) in order to have a set with the same cardinality of the smallest class.</p>
# <img src="img/undersample.png" width="400" height="500">

# In[32]:


def undersampling(x,y):
    tot=x.copy()
    tot["label"]=y
    yes=tot[tot["label"]==1]
    no=tot[tot["label"]==0].sample(n=len(yes),random_state=42)
    final=pd.concat([yes,no])
    label=final["label"].values.tolist()
    data=final.drop(["label"],axis=1)
    return data,label


# In[33]:


def oversampling(x,y):
    data_old=x.copy()
    label_old=y
    sm=SMOTE(sampling_strategy="minority",random_state=42)
    data,label=sm.fit_resample(data_old,label_old)
    return data,label


# In[34]:


def my_k_fold(configuration,algorithm,typology):
    f1_scores=[]
    configurations=[]
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for configuration in ParameterGrid(configuration):
        configurations.append(configuration)
        f1=[]
        for train_indices,validation_indices in skf.split(df_train,label_train):
            x_train=df_train.iloc[train_indices]
            y_train=np.array(label_train)[train_indices]
            
            if typology=="undersampling":
                x_train,y_train=undersampling(x_train,y_train)
            else:
                x_train,y_train=oversampling(x_train,y_train)
             
            x_validation=df_train.iloc[validation_indices]
            y_validation=np.array(label_train)[validation_indices]
            
            classifier=algorithm(**configuration)
            classifier.fit(x_train,y_train)
            prediction=classifier.predict(x_validation)
            fscore=f1_score(y_validation,prediction)
            f1.append(fscore)
        mean_f1=np.average(f1)
        f1_scores.append(mean_f1)
    maximum=max(f1_scores)
    if len(np.where(f1_scores==maximum)[0].tolist())>1:
        index=np.where(f1_scores==maximum)[0].tolist()[0]
    else:    
        index=np.where(f1_scores==maximum)[0].item()
    configuration=configurations[index]
    return configuration,maximum,configurations,f1_scores


# In[35]:


def plot_all(classifier,classifier_name,best_configuration,best_value,all_configurations,all_values,method):
    classifier=classifier(**best_configuration)
    if method=="undersampling":
        x_train,y_train=undersampling(df_train,label_train)
    else:
        x_train,y_train=oversampling(df_train,label_train)
    classifier.fit(x_train,y_train)
    prediction=classifier.predict(df_test)
    precision,recall,fscore,support=precision_recall_fscore_support(label_test,prediction)
    acc=accuracy_score(label_test,prediction)

    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    for i in range(len(prediction)):
        if prediction[i]==1:
            if label_test[i]==1:
                true_positive=true_positive+1
            else:
                false_positive=false_positive+1
        else:
            if label_test[i]==0:
                true_negative=true_negative+1
            else:
                false_negative=false_negative+1
    altosinistra=true_negative/(false_positive+true_negative)
    altodestra=1-altosinistra
    bassodestra=true_positive/(false_negative+true_positive)
    bassosinistra=1-bassodestra
    z=np.array([bassosinistra,bassodestra,altosinistra,altodestra])
    z.resize(2,2)
    z_text = np.around(z, decimals=2) # Only show rounded value (full value on hover)
    
    str_precision=str(round(precision[0],3))+" - "+str(round(precision[1],3))
    str_recall=str(round(recall[0],3))+" - "+str(round(recall[1],3))
    str_fscore=str(round(fscore[0],3))+" - "+str(round(fscore[1],3))
    str_accuracy=str(round(acc,3))

    values=[["Precision","Recall","Fscore","Accuracy"],[str_precision,str_recall,str_fscore,str_accuracy]]
    
    all_c=[]
    all_v=[]
    for i in range(len(all_configurations)):
        all_c.append(str(all_configurations[i]))
        all_v.append(all_values[i])
    values_all_configuration=[all_c,all_v]
    

    fig1 = ff.create_annotated_heatmap(z,x=[0,1],y=[1,0],annotation_text=z_text, colorscale='Blues',hoverinfo='z')
    for i in range(len(fig1.data)):
        fig1.data[i].xaxis='x1'
        fig1.data[i].yaxis='y1'
    fig1.layout.xaxis1.update(side="bottom",title="Predicted values")
    fig1.layout.yaxis1.update(side="bottom",title="True values")
    
    fig2 = go.Figure(data=[go.Table(cells=dict(values=["RESULTS OBTAINED WITH THE BEST CONFIGURATION FOUND, metrics summarization and confusion matrix"],line_color='white',
               fill_color='white',))])
     
    

    if len(all_configurations)==0:
        fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[   [{"type": "table"}],[{"type": "scatter"}],[{"type": "table"}],[{"type": "table"}],
                   ],
        )
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metrics","Non Default - Default"],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=values,
                    align = "left")
            ),
            row=1, col=1
        )
        fig.add_trace(
            fig1.data[0],
            row=2, col=1
        )
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"{classifier_name} Results",
        )
        fig.layout.update(fig1.layout)

        fig.show()
        return recall
        

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"type": "table"}],
               [{"type": "table"}],
               [{"type": "table"}],
               [{"type": "scatter"}],
               ],
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=["All configurations tried","Fscore"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=values_all_configuration,
                align = "left")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Best configuration","Fscore"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[str(best_configuration),best_value],
                align = "left")
        ),
        row=2, col=1
    )
    fig.add_trace(
        fig2.data[0],
        row=3, col=1
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metrics","Non Default - Default"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=values,
                align = "left")
        ),
        row=4, col=1
    )
    fig.add_trace(
        fig1.data[0],
        row=5, col=1
    )
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"{classifier_name} Results",
    )
    fig.layout.update(fig1.layout)
    fig.layout.update(fig2.layout)

    fig.show()
    return recall


# ### Random Forest - undersampling <a name="randomforest"></a>
# <p>Random forest is an esemble of decision trees<br>
#     <img src="img/random forest.png"  width="400" height="500">
# <p><b>Random forest construction:</b>
#     <ul>
#         <li>$B$ decision trees are built at training time.</li>
#         <li>Each tree is trained on a random sample with replacement of the training set $X = x_{1}, ..., x_{n}$, considering only a random set of features. In this way are obtained different samples $D_{1} ... D_{j} ... D_{B}$, one for each tree.<br>
#             So unlike in bagged trees where each tree is trained on all the available features,
#             in random forest each tree can be trained only on $m$ predictors. <br>
#             Usually $m=\sqrt p$, where $p$ is the total number of predictors.<br>
#             This tweak is important since decorrelates the trees, in fact if there is a strong predictor all the classifiers will use it and consequently all the trees will be very similar to each other.</li></ul></p>
# <p><b>Random forest classification:</b>
#     <ul>
#         <li>The class is assigned by majority voting by all the trees.</li></ul></p>
# <p><b>Strong and weak point:</b>
#    <ul>
#         <li>Improve accuracy and stability (robust to noise and outliers).</li>
#         <li>Avoid overfitting.</li>
#        <li>The result could be difficult to interpret since each prediction is given by hundreds of trees.</li>
#     </ul>
# </p>
# <p><b>Hyperparameters selection:</b>
# <ul>
#     <li>To find the best configuration it's performed a k-fold cross validation, the hyperparameters tuned are:
#         <ul>
#             <li><b>criterion</b>:
#                 <ul>
#                     <li>It's a function used to measure the quality of a split.</li>
#                     <li><b>gini</b> is for Gini impurity:
#                         <ul>
#                             <li>It's chosen the split position that has the least gini index</li>
#                             <li>$GINI_{split}=\sum_{i=1}^k \frac{n_{i}}{n}GINI(i)$</li>
#                             <li>$GINI(t)=1-\sum_{j}[p(j|t)]^2$<br>
#                             where $p(j|t)$ is the relative frequency of class $j$ at node $t$</li>
#                         </ul>
#                     </li>
#                     <li><b>entropy</b> is for information gain
#                         <ul>
#                             <li>It chooses the split that maximizes GAIN</li>
#                             <li>$GAIN_{split}=Entropy(p)-( \sum_{i=1}^{k}\frac{n_{i}}{n}Entropy(i) )$<br> 
#                             where $p$ is split into $k$ partitions, $n_{i}$ is the number of records in partition $i$</li>
#                             <li>$Entropy(t)= \sum_{j} p(j|t)log_{2}p(j|t)$</li>
#                         </ul>
#                     </li>
#                 </ul>
#             </li>
#             <li><b>max_depth</b>
#                 <ul>
#                     <li>It's the maximum depth of the tree.</li>
#                     <li><b>None</b> indicates the nodes are expanded until all leaves are pure.</li>
#                     <li><b>10</b> indicates the maximum depth of the tree can be 10 leaves.</li>
#                 </ul>
#             </li>
#              <li><b>n_estimators</b>
#                 <ul>
#                     <li>The number of trees in the forest.</li>
#                 </ul>
#             </li>
#         </ul>            
#     </li>
# </ul>

# In[45]:


param_grid_random_forest={"n_estimators":[100,300,500],"criterion":["gini","entropy"],"max_depth":[None,10]}
configuration_random_forest,best_rf,configurations_rf,f1_scores_rf=my_k_fold(param_grid_random_forest,RandomForestClassifier,"undersampling")


# In[77]:


recall_random_forest_undersampling=plot_all(RandomForestClassifier,"Random Forest",configuration_random_forest,best_rf,configurations_rf,f1_scores_rf,"undersampling")


# ### LOGISTIC REGRESSION - undersampling <a name="logisticregression"></a>
# <p>Logistic regression is a statistical method for predicting binary classes, namely it's a special case of linear regression where the targets are categorical and only two classes are present: response variable $Y$ is Bernoulli (if there are more classes it's the so called <i>multinomial logistic regression</i>) .</p>
# <p>The difference with linear regression is that <b>Sigmoid function</b> is used as cost function in order to limit it between 0 and 1.<br>
#     The Sigmoid function is used to map predictions to probabilities.<br>
#     $f(X)=\frac{1}{1+e^{-(\beta_{0}+\beta_{1}X)}}$<br>
# <p>Then the log odds are computed and is obtained the following equation:<br>
#     $log \frac{f(X)}{1-f(X)}= \beta_{0}+\beta_{1}X$<br>
# <p>In order to minimize the error,$\beta_{i}$ coefficients are estimated through the Maximum Likelihood Estimation<br>
#     $l(\beta_{0},\beta_{1})=\prod_{i:y_{i}=1}p(x_{i})\prod_{i:y_{i}=0}(1-p(x_{i}))$<br><br>
#     Once $\beta_{i}$ have been estimated, it's enough to compute the probabilities, and the record is assigned to the class with the highest probability.</p>
#     
# <p>What mentioned above is about problems where there is a single predictor, instead if there are more predictors the log odds have to be generalized:<br>
# $log \frac{f(X)}{1-f(X)}= \beta_{0}+\beta_{1}X+...+\beta_{p}X_{p}$<br>
# since $f(X)=\frac{1}{1+e^{-(\beta_{0}+\beta_{1}X+...+\beta_{p}X_{p})}}$<br> </p>
#    
# <p><b>Hyperparameters selection:</b>
# <ul>
#         <li>To find the best configuration it's performed a k-fold cross validation, the hyperparameters tuned are:
#             <ul>
#                 <li><b>C</b>:
#                 <ul><li>It's the inverse of regularization strength, smaller C stronger the regularization is.</li></ul>
#                 <li><b>tol</b>:
#                     <ul><li>Tolerance for stopping criteria.</li>                  

# In[36]:


param_grid_logistic_regression={"penalty":["l2"],"C":[0.1,1,10,100],"tol":[1e-5,1e-4,1e-3]}
configuration_logistic_regression,best_lr,configurations_lr,f1_scores_lr=my_k_fold(param_grid_logistic_regression,LogisticRegression,"undersampling")


# In[37]:


recall_logistic_regression_undersampling=plot_all(LogisticRegression,"Logistic Regression",configuration_logistic_regression,best_lr,configurations_lr,f1_scores_lr,"undersampling")


# ### SUPPORT VECTOR MACHINE - undersampling <a name="svm"></a>
# <p>Support Vector Machine aims to find the hyperplane that separates in the best way the classes in the feature space.<br>
# The general equation for a hyperplane is:<br>
# $\beta_{0}+\beta_{1}X_{1}+...+\beta_{p}X_{p}=0 $</p>
# 
# <p><b>HARD MARGIN</b><br>
# <ul><li>It's the basic version of SVM, suitable when data are linear separable.
#     <li>It looks for the hyperplane that makes the biggest margin between two classes.</li>
#     <li>$maximize_{\beta_{0}, \beta_{1},...,\beta_{p}}M$ subjects to 
#         <ul><li>$\sum_{j=1}^{p}\beta_{j}^{2}=1$</li>
#             <li>$y_{i}(\beta_{0}+\beta_{1}x_{i1}+...+\beta_{p}x_{ip})\ge M$</li></ul></li></ul>
# <img src="img/hard margin.png"  width="300" height="500">
# <b>SOFT MARGIN</b><br>
# <ul><li>When there are a lot of noisy hard margin is not able to work well, for this reason in these cases a more suitable version of SVM is the one which allows some mistakes</li>
#     <li>$maximize_{\beta_{0}, \beta_{1},...,\beta_{p},\epsilon_{1},...,\epsilon_{n}}M$ subjects to 
#         <ul><li>$\sum_{j=1}^{p}\beta_{j}^{2}=1$</li>
#             <li>$y_{i}(\beta_{0}+\beta_{1}x_{i1}+...+\beta_{p}x_{ip})\ge M(1-\epsilon_{i})$</li>
#             <li>where $\epsilon_{i}\ge0,\sum_{i=1}^{n}\epsilon_{i}\le C$</li>
#             <li>C is a regularization parameter, smaller is C smaller will be the margin; so bigger C more the model will be allowed to do mistakes</li></ul></li></ul>
# <img src="img/soft margin.png"  width="250" height="300">
# <b>KERNEL TRICK</b>
# <ul><li>Sometimes can happen that independently of C linear boundary won't work:</li>
# <img src="img/kernel.png"  width="260" height="300">
# <li>So if data are not linearly separable in the starting dimensional space, it can be linearly separable in a higher dimensional space: <b>kernel trick</b>.</li>
#     <li>A mapping function that maps data into an higher dimensional space could be very complex and computationally heavy.</li>
#     <li>To overcome this problem, kernel function could be very useful.
#     <ul><li>In fact since the classifier can be represented as a sum of inner products:<br>
#         $f(x)=\beta_{0}+\sum_{i=1}^{n}\alpha_{i}\langle x{,}x_{i}\rangle$</li>
#         <li>It's useful to exploit the so called <b>kernel functions</b> that can be easily compute the inner-products between observations:<br>
#     <ul>
#         <li><b>Polynomial kernel</b>: $K(x_{i},x_{i'})=(1+\sum_{j=1}^{p}x_{ij}x_{i'j})^{d}$</li>
#         <li><b>Radial Basis function kernel</b>: $K(x_{i},x_{i'})=e^{-\gamma||x_{i}-x_{i'}||}$</li>
#     </ul></li></ul>
#     <li>The final result will be non-linear decision boundaries in the original space.</li>
#     <ul><li>If rbf kernel is used the resulting boundaries are:
# <img src="img/rbf.png"  width="250" height="300">    </li>
#     <li>If polynomial kernel is used (degree=3) the resulting booundaries are:
#     <img src="img/poly.png"  width="250" height="300"> </li></ul>
#     </li></ul>
# 
# <br>
# <p><b>SVM</b> is optimal in terms of minimizing the risk of making mistakes:
# <ul><li>Minimizes the empirical risk:</li>
#         <ul><li>It's the risk of making mistakes on trainig data.</li>
#             <li>$\widehat{R}=\frac{1}{m}\sum_{i}^{m}L(h(x_{i}),y_{i})$</li>
#             <ul><li>$L$ is the loss function (the cost of predicting $\widehat{y}$ instead of $y$)</li>
#                 <li>$h$ is the studied hypothesis</li>
#             <li>$m$ is the cardinality of the sample</li></ul></ul>
#         <li>Minimizes the structural risk:
#             <ul><li>It's the risk of making mistakes on new data, it's the best approximation for the Bayes risk (the best achievable performance).</li>
#                 <li>$R=\frac{1}{m}\sum_{i}^{m}L(h(x_{i}),y_{i})+penalty(H_{n},m)$</li>
#                 <ul><li>Penalty allows to specify the tradeoff between accuracy and capacity of the model to be flexible with new data.</li></ul></ul></ul>
# <br>
# <p><b>Hyperparameters selection:</b>
#     <ul>
#         <li>To find the best configuration it's performed a k-fold cross validation, the hyperparameters tuned are:
#             <ul>
#                 <li><b>kernel</b>:
#                 <ul><li>It's the kernel type used by the algorithm.</li>
#                     <li>Three different kernels are tried:
#                         <ul><li><b>Linear</b>, to implement a soft margin SVM</li>
#                             <li><b>Rbf</b>, to implement SVM with <i>Radial Basis Function</i> kernel</li>
#                     <li><b>Poly</b>, to implement <i>Polynomial Function</i> kernel</li></ul></ul></li>
#                 <li><b>C</b>:
#                     <ul><li>It indicates the trade off between misclassification of training examples against simplicity of the decision surface.</li>
#                         <li>Low C makes the decision surface smoother.</li></ul>
#                 <li><b>gamma</b>:
#                     <ul><li>To use only when kernel is <i>poly</i> or <i>rbf</i>.</li>
#                         <li>It is the influence of a single training example.</li>
#                         <li>The larger is gamma, the closer other examples must be to be affected.</li>
#                         <li>Two different values are tuned:</li>
#                         <ul><li><b>auto</b>: $gamma=\frac{1}{\#features}$</li>
#                             <li><b>scale</b>:$gamma=\frac{1}{\#features * X.var()}$</li></ul></ul>
#                         <li><b>degree</b>:</li>
#                 <ul><li>Only when kernel is <i>poly</i></li>
#                     <li>It indicates the degree of the polynomial kernel function</li>
#                     <li>In order to avoid overfitting are tried only values: 2 and 3.</li></ul></ul>
#                 

# #### Linear SVM - undersampling<a name="linearsvm"></a>

# In[49]:


param_grid_linear_SVM={"kernel":["linear"],"C":[0.1,1,10]}
configuration_linear_SVM,best_linear_SVM,configurations_svm,f1_scores_svm=my_k_fold(param_grid_linear_SVM,SVC,"undersampling")


# In[83]:


recall_linear_SVM_undersampling=plot_all(SVC,"Linear SVM",configuration_linear_SVM,best_linear_SVM,configurations_svm,f1_scores_svm,"undersampling")


# #### Rbf SVM - undersampling<a name="rbfsvm"></a>

# In[51]:


param_grid_RBF_SVM={"kernel":["rbf"],"C":[0.1,1,10],"gamma":["auto","scale"]}
configuration_RBF_SVM,best_svm_rbf,configurations_svm_rbf,f1_scores_svm_rbf=my_k_fold(param_grid_RBF_SVM,SVC,"undersampling")


# In[84]:


recall_RBF_SVM_undersampling=plot_all(SVC,"Rbf SVM",configuration_RBF_SVM,best_svm_rbf,configurations_svm_rbf,f1_scores_svm_rbf,"undersampling")


# #### Poly SVM - undersampling<a name="polysvm"></a>

# In[53]:


param_grid_POLY_SVM={"kernel":["poly"],"degree":[2,3],"C":[0.1,1,10],"gamma":["auto","scale"]}
configuration_POLY_SVM,best_svm_poly,configurations_svm_poly,f1_scores_svm_poly=my_k_fold(param_grid_POLY_SVM,SVC,"undersampling")


# In[85]:


recall_POLY_SVM_undersampling=plot_all(SVC,"Poly SVM",configuration_POLY_SVM,best_svm_poly,configurations_svm_poly,f1_scores_svm_poly,"undersampling")


# ### Linear Discriminant Analysis - undersampling <a name="lda"></a>
# <p>LDA, Linear Discriminant Analysis, is the generalization of Fisher's Discriminant Analysis.</p>
# <p>It's similar to Logistic regression, but instead of modeling $Pr(Y=k|X=x)$ using the logistic function, LDA computes the distribution of the predictors X separately for each Y class, and then thank to Bayes' theorem it finds $Pr(Y=k|X=x)$.</p>
# 
# <p>It looks for finding a linear combination of features that separates two or more classes, events. The resulting combination can be used as a linear classifier or for dimensionality reduction.</p>
# <p>It's based on some assumptions:<br>
# <ul><li><b>Multivariate normality</b>: Independent variables are normal.</li>
#     <li><b>Homoscedasticity</b>: Variances among group variables are the same across levels of predictors.</li></ul></p>
# <p>Differently from Gaussian NB, no independence assumptions are done over the features.</p>
# <p>Example:<br>
#     Assuming that it's known how a n dimensional vector X vaires in two populations: <i>Y</i> (stands for default), <i>N</i> (for no default):<br>
# <ul><li>in <i>Y</i> population: $X\sim N(\mu_{Y},\Sigma_{Y})$</li>
# <li>in <i>N</i> population: $X\sim N(\mu_{N},\Sigma_{N})$</li>
#     <li>When a random person is selected, the purpose is to understand if he belongs to <i>Y</i> or <i>N</i> class, by analyzing his X.</li>
#     <li>To solve this it's possible to look at the likelihood ratio test:<br>
#     If $\frac{f(x;\mu_{Y},\Sigma_{Y})}{f(x;\mu_{N},\Sigma_{N})} \gt t$ then the person is assigned to <i>Y</i> class.</li>
#     <li>By resolving the previous formula it's found:<br>
#     $(x-\mu_{N})'\Sigma_{N}^{-1}(x-\mu_{N})-(x-\mu_{Y})'\Sigma_{Y}^{-1}(x-\mu_{Y}) \gt t^{*}$ </li>
#     <ul><li>Where $(x-\mu_{N})'\Sigma_{N}^{-1}(x-\mu_{N})$ and $(x-\mu_{Y})'\Sigma_{Y}^{-1}(x-\mu_{Y})$ are the Mahalanobis distances of X from the means of the two classes.</li></ul>
#     <li>Because of the <i>Homoscedasticity</i> assumption $\Sigma_{Y}=\Sigma_{N}$ and so the final formulation is:<br>
# $x'\Sigma^{-1}(\mu_{Y}-\mu_{N})\gt t^{**}$</li></ul></p>
#     

# In[55]:


param_grid_LDA={"tol":[1e-5,1e-4,1e-3,1e-2]}
configuration_LDA,best_lda,configurations_lda,f1_scores_lda=my_k_fold(param_grid_LDA,LinearDiscriminantAnalysis,"undersampling")


# In[86]:


recall_LDA_undersampling=plot_all(LinearDiscriminantAnalysis,"Linear Discriminant Analysis",configuration_LDA,best_lda,configurations_lda,f1_scores_lda,"undersampling")


# ### Gaussian NB - undersampling <a name="gaussian"></a>
# <p>This is an algorithm which applies Bayes' theorem with strong feature independence assumptions.</p>
# <br>
# In other words, Gaussian Naive Bayes classifier assumes that the effect of a particular feature is independent of other features, and evenif they are interdependent, they will be considered as independent.<br>
# From a mathematical point of view this algorithm is based on the formula:<br>
# $P(A|B)=\frac{P(B|A)P(A)}{P(B)}$<br>
# A simplified example to understand how this algorithm works could be:<br>
# Knowing that the marriage state is equal to single, it's interesting to calculate the probability of default: <b>P(default | single)</b>.
#     <ul><li>First, the prior probabilities have to be computed, <b>P(single)</b> and <b>P(default)</b>.</li>
#     <li>Then it has to be computed <b>P(single|default)</b>.</li>
#     <li>Knowing all these probailities it's possible to compute <b>P(default|single)</b>:<br>
#     $P(default|single)=\frac{P(single|default)P(default)}{P(single)}$</li>
#     <li>Then it's computed <b>P(not default|single)</b>, in the same way.</li>
#     <li>If <b>P(default|single)</b> is greater than <b>P(not default|single)</b></li> the default will be more likely.</ul>
# 
# <br>
# <p><b>Strong and weak points:</b>
# <ul><li>If features are independent it reaches very good performances.</li>
#     <li>It's very fast on large datasets.</li>
#     <li>If features are not independent, this algorithm won't work very well.</li>
#     <li>If a class has no occurrences, or a small number of them, then the estimated posterior probability will be zero, or close to that value, and so the model is unable to make a prediction. </li></ul></p>

# In[87]:


recall_GAUSSIAN_undersampling=plot_all(GaussianNB,"Gaussian NB",{},{},{},{},"undersampling")


# ### Oversampling <a name="oversampling"></a>
# <p>The second technique used to deal with the unbalance dataset is oversampling, in particular <b>SMOTE</b>.</p>
# <img src="img/over sample.png" width="400" height="500">
# <p>SMOTE consists in creating synthetic points from the smallest class in order to obtain a balancing between the minority and majority class.<br>
#     <ul>
#         <li>It picks the distances between the points belonging to the minority class.</li>
#         <li>Along these distances new synthetic points are created</li>
#      </ul>
# <p><b>Explanation</b>
# <img src="img/smote.png" width="400" height="500">
#  <ul>
#     <li>Considering the sample $x_{i}$ to generate a new point, we have to look at also its k-nearest neighbors: the 3-nearest neighbors are included in the blue circle as illustrated in the figure above.</li>
#      <li>One of the 3-nearest neighbors is selected, for example $x_{zi}$, and $x_{new}$ is generated according to the formula:<br>
#      $x_{new}=x_{i}+\lambda \times (x_{zi}-x_{i}) $<br>
#      where $\lambda \in [0,1]$ is a random number.</li></ul></p>

# ### Random Forest - oversampling<a name="randomforestover"></a>

# In[58]:


param_grid_random_forest={"n_estimators":[100,300,500],"criterion":["gini","entropy"],"max_depth":[None,10]}
configuration_random_forest_over,best_rf_over,configurations_rf_over,f1_scores_rf_over=my_k_fold(param_grid_random_forest,RandomForestClassifier,"oversampling")


# In[88]:


recall_random_forest_oversampling=plot_all(RandomForestClassifier,"Random Forest",configuration_random_forest_over,best_rf_over,configurations_rf_over,f1_scores_rf_over,"oversampling")


# ### Logistic Regression - oversampling <a name="logisticregressionover"></a>

# In[60]:


param_grid_logistic_regression={"penalty":["l2"],"C":[0.1,1],"tol":[1e-5,1e-4,1e-3],"max_iter":[500]}
configuration_logistic_regression_over,best_lr_over,configurations_lr_over,f1_scores_lr_over=my_k_fold(param_grid_logistic_regression,LogisticRegression,"oversampling")


# In[89]:


recall_logistic_regression_oversampling=plot_all(LogisticRegression,"Logistic Regression",configuration_logistic_regression_over,best_lr_over,configurations_lr_over,f1_scores_lr_over,"oversampling")


# ### Support Vector Machine - oversampling <a name="svmover"></a>

# #### Linear SVM - oversampling <a name="linearsvmover"></a>

# In[62]:


param_grid_linear_SVM={"kernel":["linear"],"C":[0.1,1,10],"max_iter":[500000]}
configuration_linear_SVM_over,best_linear_over,configurations_linear_over,f1_scores_linear_over=my_k_fold(param_grid_linear_SVM,SVC,"oversampling")


# In[90]:


recall_linear_SVM_oversampling=plot_all(SVC,"Linear SVM",configuration_linear_SVM_over,best_linear_over,configurations_linear_over,f1_scores_linear_over,"oversampling")


# #### Rbf SVM - oversampling <a name="rbfsvmover"></a>

# In[91]:


param_grid_RBF_SVM={"kernel":["rbf"],"C":[0.1,1,10],"gamma":["auto","scale"],"max_iter":[500000]}
configuration_RBF_SVM_over,best_RBF_over,configurations_RBF_over,f1_scores_RBF_over=my_k_fold(param_grid_RBF_SVM,SVC,"oversampling")


# In[92]:


recall_RBF_SVM_oversampling=plot_all(SVC,"Rbf SVM",configuration_RBF_SVM_over,best_RBF_over,configurations_RBF_over,f1_scores_RBF_over,"oversampling")


# #### Poly SVM - oversampling<a name="polysvmover"></a>

# In[93]:


param_grid_POLY_SVM={"kernel":["poly"],"degree":[2,3],"C":[0.1,1,10],"gamma":["auto","scale"],"max_iter":[500000]}
configuration_POLY_SVM_over,best_POLY_over,configurations_POLY_over,f1_scores_POLY_over=my_k_fold(param_grid_POLY_SVM,SVC,"oversampling")


# In[94]:


recall_POLY_SVM_oversampling=plot_all(SVC,"Poly SVM",configuration_POLY_SVM_over,best_POLY_over,configurations_POLY_over,f1_scores_POLY_over,"oversampling")


# ### Linear Discriminant Analysis - oversampling<a name="ldaover"></a>
# <p>LDA is the generalization of Fisher's Discriminant Analysis.</p>

# In[95]:


param_grid_LDA={"tol":[1e-5,1e-4,1e-3,1e-2]}
configuration_LDA_over,best_lda_over,configurations_lda_over,f1_scores_lda_over=my_k_fold(param_grid_LDA,LinearDiscriminantAnalysis,"oversampling")


# In[96]:


recall_LDA_oversampling=plot_all(LinearDiscriminantAnalysis,"Linear Discriminant Analysis",configuration_LDA_over,best_lda_over,configurations_lda_over,f1_scores_lda_over,"oversampling")


# ### Gaussian NB - oversampling<a name="gaussianover"></a>

# In[97]:


recall_GAUSSIAN_oversampling=plot_all(GaussianNB,"Gaussian NB",{},{},{},{},"oversampling")


# ## Conclusion <a name="conclusion"></a>

# In[98]:


def plot_result(yes,nomi,kind):
    x=np.arange(len(nomi))    
    y_yes=[]
    y_no=[]
    for el in yes:
        y_yes.append(el[1])
        y_no.append(el[0])
    

    width = 0.35

    fig,ax=plt.subplots(figsize=(20,8))
    rects1 = ax.bar(x - width/2, y_no, width, label="TRUE NEGATIVE",color="green")
    rects2 = ax.bar(x + width/2, y_yes, width, label="TRUE POSITIVE",color="red")
    

    ax.set_title(f"Recall - {kind}",fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(nomi,fontsize=18)
    ax.legend(loc=1)


    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),textcoords="offset points",ha='center', va='bottom',fontsize=15)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    ax.grid(False)

    plt.show()


# In[99]:


undersampling_list=[recall_random_forest_undersampling,recall_logistic_regression_undersampling,recall_linear_SVM_undersampling,recall_RBF_SVM_undersampling,recall_POLY_SVM_undersampling,recall_GAUSSIAN_undersampling,recall_LDA_undersampling]
oversampling_list=[recall_random_forest_oversampling,recall_logistic_regression_oversampling,recall_linear_SVM_oversampling,recall_RBF_SVM_oversampling,recall_POLY_SVM_oversampling,recall_GAUSSIAN_oversampling,recall_LDA_oversampling]
name=["random_forest","logistic_regression","linear_SVM","RBF_SVM","POLY_SVM","GAUSSIAN","LDA"]


# In[100]:


plot_result(undersampling_list,name,"UNDERSAMPLING")


# In[101]:


plot_result(oversampling_list,name,"OVERSAMPLING")


# <p>Since the goal of the study is the one of finding the best performance model, namely the one which is able to correctly detect possible credit card default, F1 scores are taken into consideration.<br><br>
# The highest True Negative rate (no default correctly classified) is reached from <b>Random Forest with oversampling</b>: <i>0.90</i>, but the corresponding True Positive rate (default correctly classified) is very low: <i>0.48</i>. And this is not good for the bank because since the algorithm is not very good in detecting default, the financial loss will be high.<br><br>
#     The highest True Positive rate is reached from <b>Gaussian NB with oversampling</b>: <i>0.60</i>, but the corresponding True Negative rate is not satisfactory: <i>0.70</i>, this means that a lot of non default clients will be classified as default clients and so may the bank won't grant them a loan, and so they will change bank: also this is a financial loss.<br><br>
# Of course the final decision should be taken with the bank, because it knows what is the best proportion between True Positive and True Negative rates, but in my opinion a good classifier could be <b>Random Forest with undersampling</b>, <i>(tnr=0.85,tpr=0.56)</i>, or <b>SVM with rbf kernel and oversampling</b>, <i>(tnr=0.80,tpr=0.56)</i>. <br>
#     In fact both of them have good performance in terms of true negative and true positive rates.</p>

# ## References <a name="references"></a>

# [1]. "An Introduction to Statistical Learning, with Applications in R". James, Witten, Hastie, Tibshirani
# 
# [2]. [https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
# 
# [3]. [https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)
# 
# [4]. [https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html)
# 
# [5]. [https://en.wikipedia.org/wiki/Isolation_forest](https://en.wikipedia.org/wiki/Isolation_forest)
# 
# [6]. [https://scikit-learn.org/stable/supervised_learning.html#supervised-learning](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
# 
# [7]. [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)
# 
# [8]. [https://medium.com/analytics-vidhya/smote-nc-in-ml-categorization-models-fo-imbalanced-datasets-8adbdcf08c25](https://medium.com/analytics-vidhya/smote-nc-in-ml-categorization-models-fo-imbalanced-datasets-8adbdcf08c25)
# 
