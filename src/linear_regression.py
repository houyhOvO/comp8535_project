import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

with open('../dataset/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Make sure the data is writable
X_train = np.copy(X_train)
y_train = np.copy(y_train)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# save the model
with open('../models/linear_regression.pkl', 'wb') as f:
    pickle.dump(linear_regression, f)

y_pred = linear_regression.predict(X_test)

plt.scatter(y_test,y_pred)
plt.title('Actual vs Predicted (Linear Regression)')
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.savefig('../figs/linear_regression.png')
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)
