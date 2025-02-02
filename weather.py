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
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True, cmap="coolwarm", fmt=".2f")
plt.title("correlation heatmap")
plt.show()

plt.figure(figsize=(15, 10))
for i, column in enumerate (['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
 plt.subplot(3, 3, i)
 sns.boxplot(data[column])
 plt.title(f"boxplot of {column}")

plt.tight_layout()
plt.show()

#drop highly correlated column
data=data.drop(columns=['maxtemp', 'temparature', 'mintemp'], errors='ignore')

data.head()

data["rainfall"].value_counts()

#separate majority and minority class
df_majority = data[data["rainfall"]==1]
df_minority = data[data["rainfall"]==0]

print(df_majority.shape)
print(df_minority.shape)

df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_majority_downsampled.shape

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_downsampled.head()

# shuffle the final dataframe
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

df_downsampled.head()

df_downsampled["rainfall"].value_counts()

#split feature and target as x and y
x= df_downsampled.drop(columns=["rainfall"])
y= df_downsampled["rainfall"]

#splitting the data into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#model building
rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# hyperparameter tuning using gridsearchcv
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

grid_search_rf.fit(x_train, y_train)

