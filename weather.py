import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

data= pd.read_csv('Rainfall.csv')
data.shape

data.head()
data.tail()

data["day"].unique()
print("data info:")
data.info()
data.columns

#to remove empty space from column name
data.columns =data.columns.str.strip()
data.info()

# to drop day columns( because this is not required to us)
data =data.drop(columns=["day"])