import pickle
import numpy as np

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

with open('../dataset/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Make sure the data is writable
X_train = np.copy(X_train)
y_train = np.copy(y_train)

decision_tree = DecisionTreeRegressor(random_state=0)
decision_tree.fit(X_train, y_train)

# Save the model
with open('../models/decision_tree.pkl', 'wb') as f:
    pickle.dump(decision_tree, f)

y_pred = decision_tree.predict(X_test)

plt.scatter(y_test,y_pred)
x = np.linspace(0, 50, 50)
plt.plot(x, x)
plt.title('Actual vs Predicted (Decision Tree)')
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.savefig('../figs/decision_tree.png')
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)
