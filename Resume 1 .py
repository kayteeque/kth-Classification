#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from datascience import *

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.simplefilter('ignore', FutureWarning)

import otter
grader = otter.Notebook()


# # K-Nearest Classification

# Using a dataset of students that live in the Bay Area, Id like to classify students as either attendees of UC Berkeley or Leland Stanford Junior College. We have access to longitude and latitute of each given students residence during attendance as well as their corresponding college. 

# # Visualization

# In[5]:


coordinates = Table.read_table('coordinates.csv')
coordinates


# Lets inspect our data visually

# In[7]:


coordinates.scatter('longitude', 'latitude', group='school')


# In[9]:


colors = {'Berkeley':'blue','Stanford':'red'}
t = Table().with_columns('lat', coordinates.column(0),
                        'lon', coordinates.column(1),
                        'color', coordinates.apply(colors.get, 2)
                        )
Circle.map_table(t, redius=5, fill_opacity=1)


# # Our Algorithm

# Lets begin our algorithm. We will first define a function that takes in two arrays and returns the Euclidean Distance between them. 

# In[10]:


def distance(arr1, arr2):
    return np.sqrt(sum((arr2-arr1)**2))

distance_example = distance(make_array(1, 2, 3), make_array(4, 5, 6))
distance_example


# Lets now split the data into two datasets. One for training and one for testing. This will be done randomly.

# In[11]:


shuffled_table = coordinates.sample(with_replacement=False)
train = shuffled_table.take(np.arange(0,75))
test = shuffled_table.take(np.arange(75,100))

print("Training set:\t",   train.num_rows, "examples")
print("Test set:\t",       test.num_rows, "examples")
train.show(5), test.show(5);


# Lets pull out our features.

# In[12]:


features = make_array('latitude', 'longitude')
features


# Below are our functions to classify our data points based on their nearest neighbors. 

# In[13]:


def row_to_array(row, features):
    arr = make_array()
    for feature in features:
        arr = np.append(arr, row.item(feature))
    return arr

def classify(row, k, train):
    test_row_features_array = row_to_array(row, features)
    distances = make_array()
    for train_row in train.rows:
        train_row_features_array = row_to_array(train_row, features)
        row_distance = distance(train_row_features_array, test_row_features_array)
        distances = np.append(distances, row_distance)
    train_with_distances = train.with_column('Distances', distances)
    nearest_neighbors = train_with_distances.sort('Distances', descending = False).take(np.arange(0, k))
    return nearest_neighbors.group('school').column(0).item(0)
   

first_test = classify(test.row(0), 5, train)
first_test


# Lets now take a look at just a 3-nearest neighbor classifier. We will compute a proportion to determine the accuracy of our classifier. 

# In[14]:


def three_classify(row):
    return classify(row, 3, train)

test_with_prediction = test.with_column("prediction", test.apply(three_classify))
labels_correct = test_with_prediction.column('prediction')==test_with_prediction.column('school')
accuracy = np.count_nonzero(labels_correct)/test.num_rows
accuracy


# In[ ]:




