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

#to know number of missing value
data.isnull().sum()
data["windspeed"].unique()

#handaling the missing value
data["winddirection"]=data["winddirection"].fillna(data["winddirection"]).mode()[0]
data["windspeed"]=data["windspeed"].fillna(data["windspeed"].median())

data.isnull().sum()
data["rainfall"].unique()

#converting yes=1 and no=0
data["rainfall"]=data["rainfall"].map({"yes":1, "no":0})
data.head()

#exploratory data analysis
sns.set(style="whitegrid")

data.describe()
data.columns

plt.figure(figsize=(15, 10))
for i, column in enumerate (['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
 plt.subplot(3, 3, i)
 sns.histplot(data[column], kde=True)
 plt.title(f"Distribution of {column}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="rainfall", data=data)
plt.title("Distrinbution of rainfall")
plt.show()

#correlation matrix
