#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation
# 
# TASK 1:Predict the percentage of an student based on the no. of study hours.
# What will be predicted score if a student studies for 9.25 hrs/ day?

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[3]:


df.head()


# In[25]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# Let's explore the data:

# In[76]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[90]:


sns.jointplot(x='Hours',y='Scores',data=df, kind='scatter',color='tab:red')


# In[86]:


sns.pairplot(df)


# In[92]:


sns.lmplot(x='Hours',y='Scores',data=df)


# In[26]:


df.corr()


# Here,we can clearly see that there is a high correlation between hours and scores.

# # Training and Testing Data

# In[46]:


y = df['Scores']
X = df['Hours']


# In[142]:


X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# In[147]:


X.shape
y.shape
print("Shape of X :",X.shape)
print("Shape of y:",y.shape)


# Using model_selection.train_test_split from sklearn to split the data into training and testing sets:

# In[32]:


from sklearn.model_selection import train_test_split


# In[109]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[110]:


X_train.shape
y_train.shape
print("Shape of X_train :",X_train.shape)
print("Shape of y_train :",y_train.shape)


# In[111]:


X_test.shape
y_test.shape
print("shape of X_test is:",X_test.shape)
print("shape of y_test is:",y_test.shape)


# # Training the Model

# In[104]:


from sklearn.linear_model import LinearRegression


# In[112]:


lm = LinearRegression() #Creating an instance of a LinearRegression() model named lm


# Training/fitting lm on the training data:

# In[113]:


lm.fit(X_train,y_train)


# coefficients of the model:
# 
# 

# In[118]:


print('Coefficient: \n', lm.coef_)


# In[119]:


print('Intercept: \n', lm.intercept_)


# In[137]:


lm.score(X_train,y_train) #checking accuracy of the train data


# In[139]:


lm.score(X_test, y_test)  #checking accuracy of the test data


# Predicting Test Data:
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# Using lm.predict() to predict off the X_test set of the data.

# In[115]:


predictions = lm.predict( X_test)


# Creating a scatterplot of the real test values versus the predicted values:

# In[116]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[150]:


line = lm.coef_*X+lm.intercept_
plt.scatter(X, y,color="blue")
plt.plot(X, line,color="red")
plt.title('Regression line: y = mx+c')
plt.show()


# # Evaluating the Model
# 

# In[120]:


from sklearn import metrics


# In[124]:


print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('MAE:', metrics.mean_absolute_error(y_test, predictions))


# In[126]:


Actual_VsPred= pd.DataFrame({'Actual': y_test, 'Predicted': predictions })
Actual_VsPred


# In[136]:


score_result=lm.predict([[9.25]])
print(score_result)


# So, if a student studies for 9.25 hours, he/she will score 92.54% in exam.

# Conclusion: Higher the no. of hours on study, higher the score students get in exam.
