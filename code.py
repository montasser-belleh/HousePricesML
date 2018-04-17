
# coding: utf-8

# In[75]:


import pandas 


# In[76]:


data = pandas.read_csv("train.csv")


# In[77]:


data.head()


# In[78]:


df = data.dropna(thresh=len(data)-10,axis='columns') #drop all columns (axis = 1 or 'columns' ) with more than len(DATA) - 1 NaN values


# In[79]:


df.shape


# In[80]:


df.head()


# In[102]:


#Save the 'Id' column
train_ID = df['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
df.drop("Id", axis = 1, inplace = True)
df.head()


# In[81]:


y = df['SalePrice']


# In[82]:


y.head()


# In[83]:


x = df[['MSSubClass','LotArea','OverallQual','OverallCond','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']]


# In[84]:


x.head()


# In[85]:


from sklearn.tree import DecisionTreeRegressor as  DTR


# In[86]:


model = DTR()
model.fit(x,y)


# In[87]:


print(x.head())


# In[88]:


model.predict(x)


# In[89]:


#mean absolute error of decision tree
from sklearn.metrics import mean_absolute_error
pprice=model.predict(x)
error=mean_absolute_error(y,pprice)
error


# In[90]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_x, val_x, train_y, val_y = train_test_split(x, y,random_state = 0)
model = DTR()
# Fit model
model.fit(train_x, train_y)
# get predicted prices on validation data
val_predictions = model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))


# In[91]:


from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes : values from 5 -> 5000 , step is 2 
max_leaf_nodesList = []
for max_leaf_nodes in range(5,5000,2):
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    max_leaf_nodesList.append((my_mae,max_leaf_nodes))
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[92]:


max_leaf_nodesList.sort()


# In[94]:


max_leaf_nodesList #best accuracy for 73


# In[95]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_x, train_y)
pred = forest_model.predict(val_x)
h=mean_absolute_error(val_y, pred)
print(h)
#MAE using random forest


# In[96]:


#ACCURACY OF MODEL
z=(h/y.mean())
print(100-100*z)


# In[98]:


data=pandas.read_csv("test.csv")
data.head()
df=data.dropna(thresh=len(data)-10,axis=1)
#df.head()
df.dropna(axis=0)
#y=df.SalePrice
ppc=df[['MSSubClass','LotArea','OverallQual','OverallCond','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']]
predictedprices = forest_model.predict(ppc)
print(predictedprices)


# In[99]:


my_submission = pandas.DataFrame({'Id': data.Id, 'SalePrice': predictedprices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

