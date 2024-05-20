import pickle
import numpy as np
import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

with open('../dataset/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Make sure the data is writable
X_train = np.copy(X_train)
y_train = np.copy(y_train)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train, y_train)

# Save the model
with open('../models/xgboost.pkl', 'wb') as f:
    pickle.dump(xg_reg, f)

y_pred = xg_reg.predict(X_test)

plt.scatter(y_test,y_pred)
x = np.linspace(0, 50, 50)
plt.plot(x, x)
plt.title('Actual vs Predicted (XGBoost)')
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.savefig('../figs/xgboost.png')
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)
