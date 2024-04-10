import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

boston = pd.read_csv('../dataset/boston.csv')
print(boston.describe())  # Display statistics of data
print(boston.info())      # Display a summary of basic information about the data


features.append('MEDV')
correlation = boston.corr()
print(correlation)
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot()
sns.heatmap(correlation, cmap='YlGnBu', vmax=1, vmin=-1, annot=True, annot_kws={"size": 15})
plt.xticks(np.arange(len(features)) + 0.5, features)
plt.yticks(np.arange(len(features)) + 0.5, features, rotation=0)
ax.set_title('Correlation coefficient matrix heatmap')
plt.savefig('../figs/heatmap.png')
plt.show()
