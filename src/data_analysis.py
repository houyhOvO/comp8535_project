import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

boston = pd.read_csv('../dataset/boston.csv')
print(boston.describe())  # Display statistics of data
print(boston.info())      # Display a summary of basic information about the data

# data = boston.values[:, :-1]
# price = boston.values[:, -1]
# data_df = pd.DataFrame(data, columns=features)
# price_df = pd.DataFrame(price, columns=['MEDV'])


# Visualize the relationship between each pair of variables
sns.pairplot(boston)
plt.tick_params(labelsize=25)
plt.savefig('../figures/pairplot.png')


