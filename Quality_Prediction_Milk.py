#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


# ### uploading dataset using pandas.read

# In[4]:


dataset = pd.read_csv('milknew.csv')


# ### EDA

# In[5]:


dataset.head()


# In[6]:


dataset.tail()


# In[7]:


dataset.info()


# In[8]:


dataset.describe


# ### Data Preprocessing

# In[9]:


#Checking for the null values 
dataset.isnull().sum()  


# In[10]:


#Checking the unique values in Grade Column 
dataset ['Grade'].unique()


# In[11]:


#Checking the value count in the Grade column 
dataset ['Grade'].value_counts()


# ### Label Encoding 

# In[12]:


#Encoding Categorical labels with neumerical values
le = LabelEncoder()
dataset['Grade'] = le.fit_transform (dataset['Grade'])
dataset['Grade'].unique()


# ### Value of X, Y and Train_test_split function 

# In[13]:


# Setting Labels in Y & Variables in X
# Spliting the dataset for training and test in 75/25 ratio 

y = dataset["Grade"]
x = dataset.drop('Grade', axis = 1)
x_learn, x_eval, y_learn, y_eval = train_test_split(x, y, test_size=0.25, random_state=42)


# ## Model Training 
# ### These models perform best for the data we are working on
# 
# ### Naive bayes
# #### Reason for Implementing this model because of its unique characteristics and advantages 
# #### 1- Simplicity and speed: This model is computation effective, It is trained quickly with even large dataset while providing very accurate results 
# #### 2- Interpretability: It helps in decision making because the probability calculated by this model is easily understandable 
# 

# In[14]:


from sklearn.naive_bayes import GaussianNB


# In[16]:


model_nb = 'Naive Bayes'
naive_bayes = GaussianNB()

#Fiting the model with data

naive_bayes.fit(x_learn,y_learn)

#using .predict function for prediction and storing it in Naive_bayes_pred

naive_bayes_pred = naive_bayes.predict(x_eval)
naive_bayes_conf_matrix = confusion_matrix(y_eval, naive_bayes_pred)

#comparing the predtion with the actual test labels 

naive_bayes_acc_score = accuracy_score(y_eval, naive_bayes_pred)
print("confussion matrix")
print(naive_bayes_conf_matrix)
print("\n")
print("Accuracy of Naive Bayes model:",naive_bayes_acc_score*100,'\n')
print(classification_report(y_eval,naive_bayes_pred))


# ### Random Forest
# #### Reason for choosing random forest is that it provide best accuracy adn performs very good in multiple problems. In addition it is very robust in missing data, it handles missing values very efficiently
# #### 2- It provides a measure of feature importance which also helps us to understand which feature is important and which feature is not important
# #### 3- It captures the non-linear relationship between features and the target variable
# #### 4- It handles the imbalanced dataset very efficiently and provide accurate outputs (results)

# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


rand_for = RandomForestClassifier() 
params = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

grid_s = GridSearchCV(rand_for, param_grid= params, cv = 5,scoring = "roc_auc",verbose=1)
grid_s.fit(x_learn, y_learn)
print(grid_s.best_score_)
print(grid_s.best_estimator_)
print(grid_s.best_params_)


# # Confusion Matrix 
# #### Confusion matrix is a very powerful evaluation tool, It provide a detailed breakdown between the predicted and the actual values. Confusion matrix helps the developer to understand where the model is lacking and at which point or class the model needs to train more 
# #### It helps us to identify the missclassification with its actual label (value)
# #### If we are dealing with an imbalanced class it is quite possible that we are getting a good accuracy but still the model is not good, so it helps us to identify the error and optimization can be more efficient while we are using confusion matrix 
# ### Confusion Matrix - Random forest Classifier 

# In[19]:


rand_for =RandomForestClassifier(criterion='gini', max_depth=4, max_features='auto',
                       n_estimators=200)
rand_for.fit(x_learn, y_learn)
y_predict_rand_for = rand_for.predict(x_eval)

# confusion_matrix
cm = confusion_matrix(y_eval, y_predict_rand_for)
sns.heatmap(cm, annot=True, fmt="d")


# ### Accuracy Score 

# In[20]:


print('Accuracy',accuracy_score((y_eval), y_predict_rand_for) * 100)
print('Precision',precision_score(y_eval, y_predict_rand_for, average='macro') * 100)
print('Recall',recall_score(y_eval, y_predict_rand_for, average='macro') * 100)


# ### K-Nearest Neighbour
# #### KNN is a simplea and intuitive algorithm, It does not require complex calculations. It provide as our target label by measuring the distance with the nearest instances, the number of instances it check is based on the number of K. 
# #### The distance which it calculates to predict the label can be euclidean distance or manhattan distance.

# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[22]:


params = {'n_neighbors':range(1,20),"metric":["euclidean", "manhattan"],
              'algorithm' :['ball_tree','kd_tree','brute']
        
             }
knn=KNeighborsClassifier()
grid_s = GridSearchCV(knn, param_grid= params, cv = 5,scoring = "roc_auc",verbose=1)
grid_s.fit(x_learn, y_learn)
print(grid_s.best_score_)
print(grid_s.best_estimator_)
print(grid_s.best_params_)


# ### Confusion Matrix - K-Nearest Neighbour

# In[23]:


#Training KNN Model 

knn = KNeighborsClassifier(algorithm='ball_tree', metric='euclidean', n_neighbors=1)
knn.fit(x_learn, y_learn)
y_predict_knn = knn.predict(x_eval)

# confusion_matrix

cm = confusion_matrix(y_eval, y_predict_knn)
sns.heatmap(cm, annot=True, fmt="d")


# ### Accuracy Score 

# In[24]:


print('Accuracy',accuracy_score((y_eval), y_predict_knn)* 100)
print('Precision',precision_score(y_eval, y_predict_knn, average='macro') * 100)
print('Recall',recall_score(y_eval, y_predict_knn, average='macro') * 100)


# ### Decision Tree Classifier 
# #### Decision trees are very popular as they provide us the details about the model training process, It also gives us the exact reason for the split, on what basis the tree has been splited (in can be gini entropy) 
# 
# #### This helps the developer to understand more about the data and the problem .

# In[25]:


from sklearn.tree import DecisionTreeClassifier


# In[26]:


param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier(random_state=1024)
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(x_learn, y_learn)


# In[27]:


final_model = grid_search.best_estimator_
final_model


# ### Confusion Matrix - Decision Trees

# In[28]:


d_t_c =DecisionTreeClassifier(ccp_alpha=0.001, criterion='gini', max_depth=8,
                       max_features='auto', random_state=1024)
d_t_c.fit(x_learn, y_learn)
y_predict_d_t_c = d_t_c.predict(x_eval)
# confusion_matrix
cm = confusion_matrix(y_eval, y_predict_d_t_c)
sns.heatmap(cm, annot=True, fmt="d")


# ### Accuracy Score 

# In[29]:


print('Accuracy',accuracy_score((y_eval), y_predict_d_t_c) * 100)
print('Precision',precision_score(y_eval, y_predict_d_t_c, average='macro') * 100)
print('Recall',recall_score(y_eval, y_predict_d_t_c, average='macro') * 100)


# ### SVM
# #### SVM is very robust to overfitting, Overfitting is the major problem which usually the developer faces, 
# #### It is quite possible that your model  is performing very efficient with the training and testing data but when it is deployed in real world it fails to provide accurate and efficient results. 
# 
# #### It is very effective to non-linear problems

# In[30]:


from sklearn.svm import SVC


# In[31]:


param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 


# In[32]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(x_learn, y_learn)


# In[33]:


print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning

print(grid.best_estimator_)


# ### Confusion Matrix - SVC

# In[34]:


svc = SVC(kernel = 'rbf', C = 1000, gamma = 0.01)
svc.fit (x_learn, y_learn)
y_predict_svc = svc.predict (x_eval)

cm = confusion_matrix(y_eval, y_predict_svc)
sns.heatmap(cm, annot=True, fmt="d")


# ### Accuracy Score 

# In[35]:


print('Accuracy',accuracy_score((y_eval), y_predict_svc) * 100)
print('Precision',precision_score(y_eval, y_predict_svc, average='macro') * 100)
print('Recall',recall_score(y_eval, y_predict_svc, average='macro') * 100)


# ### MLP Classifier 
# #### It is an universal approximator, It can approximate any countinous function given proper training and enough neurons 
# #### Feature learning and representation: It is very efficient in learning and extracting relevant information from the raw data.
# #### Scalability: It can handle large datasets very easily and efficiently.

# In[36]:


from sklearn.neural_network import MLPClassifier


# In[37]:


param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
    'max_iter': [50, 100, 150],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}


# In[38]:


grid = GridSearchCV(MLPClassifier(), param_grid, n_jobs= -1, cv=5)
grid.fit(x_learn, y_learn)

print(grid.best_params_) 


# ### MLP Training 
# 
# ### Confusion Matrix - MLP

# In[39]:


mlp_clf = MLPClassifier(hidden_layer_sizes=(100,50,30), alpha = 0.05,
                        max_iter = 150,activation = 'tanh', learning_rate = 'constant',
                        solver = 'adam')

mlp_clf.fit(x_learn, y_learn)
y_pred_mlp = mlp_clf.predict(x_eval)

cm = confusion_matrix(y_eval, y_pred_mlp)
sns.heatmap(cm, annot=True, fmt="d")



# ### Accuracy Score 

# In[40]:


print('Accuracy',accuracy_score((y_eval), y_pred_mlp) * 100)
print('Precision',precision_score(y_eval, y_pred_mlp, average='macro') * 100)
print('Recall',recall_score(y_eval, y_pred_mlp, average='macro') * 100)


# ### Model Evaluation 
# #### Comapring each model's accuracy 

# In[42]:


model_ev = pd.DataFrame({'Model': ['MLP Classifier','Naive Bayes','Random Forest',
                    'K-Nearest Neighbour','Decision Tree','Support Vector Machine'], 'Accuracy': [ accuracy_score((y_eval), y_pred_mlp) * 100,
                    naive_bayes_acc_score*100,accuracy_score((y_eval), y_predict_rand_for) * 100,accuracy_score((y_eval), y_predict_knn)* 100,accuracy_score((y_eval), y_predict_d_t_c) * 100,accuracy_score((y_eval), y_predict_svc) * 100]})
model_ev


# ## Ensemble Learning 


# In[ ]:





# In[43]:


from mlxtend.classifier import StackingCVClassifier


# In[44]:


scv=StackingCVClassifier(classifiers=[d_t_c,knn,svc],meta_classifier= svc,random_state=42)
scv.fit(x_learn, y_learn)
scv_predicted = scv.predict(x_eval)
scv_conf_matrix = confusion_matrix(y_eval, scv_predicted)
scv_acc_score = accuracy_score(y_eval, scv_predicted)
print("confussion matrix")
print(scv_conf_matrix)
print("\n")
print("Accuracy of StackingCVClassifier:",scv_acc_score*100,'\n')
print(classification_report(y_eval,scv_predicted))


# In[ ]:





# In[ ]:





# In[ ]:




