import pandas as pd
import numpy as np
import gc
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

print('loading data')
df = pd.read_csv('Dataset.csv')

print('preprocessing data')
month_mapper = {'Dec':12, 'Jan':1, 'Oct':10, 'Jun':6, 'Feb':2, 'Mar':3, 'Aug':8, 'Apr':4, 'Jul':7, 'May':5, 'Sep':9, 'Nov':11, '0':0}
week_mapper = {'Wednesday':3, 'Friday':5, 'Saturday':6, 'Sunday':7, 'Monday':1, 'Tuesday':2, 'Thursday':4, '0':0}

df['Month'] = df['Month'].map(month_mapper)
df['DayOfWeek'] = df['DayOfWeek'].map(week_mapper)

df['MonthClaimed'] = df['MonthClaimed'].map(month_mapper)
df['DayOfWeekClaimed'] = df['DayOfWeekClaimed'].map(week_mapper)

df['MonthClaimed'] = df['MonthClaimed'].astype('int')
df['DayOfWeekClaimed'] = df['DayOfWeekClaimed'].astype('int')

df['Accident_date'] = ((df['WeekOfMonth'] - 1) * 7) + df['DayOfWeek']
df['claim_date'] = ((df['WeekOfMonthClaimed'] - 1) * 7) + df['DayOfWeekClaimed']

# preprocessing

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

df = df.dropna()

print('working on graphing')
f = df[df['FraudFound_P'] == 1]
n = df[df['FraudFound_P'] == 0]

print('doing PCA')
clf_pca_2 = PCA(n_components=3)
X_f = f.drop(columns='FraudFound_P', axis=1)
X_n = n.drop(columns='FraudFound_P', axis=1)
X_f = clf_pca_2.fit_transform(X_f)
X_n = clf_pca_2.fit_transform(X_n)

X_f[:, 1] = X_f[:, 1] ** 2
X_n[:, 0] = X_n[:, 0] ** 2

print('finally ploting')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_f[:, 0], X_f[:, 1], X_f[:, 2], c='r', marker='o')
ax.scatter(X_n[:, 0], X_n[:, 1], X_n[:, 2], c='g', marker='^')

plt.show()