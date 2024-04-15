import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


boston = pd.read_csv('../dataset/boston.csv')
print(boston.describe())  # Display statistics of data
print(boston.info())      # Display a summary of basic information about the data


correlation = boston.corr()      # Calculate correlation matrix
# print(correlation)

# Heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, cmap='YlGnBu', linewidths=0.1, annot=True, annot_kws={"size": 20, "weight":"bold"})  # Generate a heatmap for correlation
plt.title('Correlation coefficient heatmap', fontsize=20)
plt.savefig('../figs/heatmap.png')
plt.show()

# Pair plot
plt.figure(figsize=(15, 15))
sns.pairplot(boston)
plt.savefig('../figs/pairplot.png')
plt.show()
