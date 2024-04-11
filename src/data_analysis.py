import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

boston = pd.read_csv('../dataset/boston.csv')
print(boston.describe())  # Display statistics of data
print(boston.info())      # Display a summary of basic information about the data


# features.append('MEDV')
correlation = boston.corr()      # Calculate correlation matrix
print(correlation)
plt.figure(figsize=(15, 15))
# ax = fig.add_subplot()           # Add a subplot
sns.heatmap(correlation.iloc[:, 13:], cmap='YlGnBu', linewidths=0.1, annot=True, annot_kws={"size": 20, "weight":"bold"})  # Generate a heatmap for correlation
# plt.xticks(np.arange(len(features)) + 0.5, features)
# plt.yticks(np.arange(len(features)) + 0.5, features, rotation=0)
# plt.xticks(np.arange(13)+0.5, features)
# plt.yticks(np.arange(13)+0.5, features)
# ax.set_title('Correlation coefficient matrix heatmap')
plt.title('Correlation coefficient heatmap', fontsize=20)
plt.savefig('../figs/heatmap.png')
plt.show()
