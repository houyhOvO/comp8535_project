import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from update_model_results import update_model_results

# 加载数据
with open('../dataset/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# 确保数据可写
X_train = np.copy(X_train)
y_train = np.copy(y_train)

# 创建并训练随机森林模型
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# 保存模型
with open('../models/random_forest_regressor.pkl', 'wb') as f:
    pickle.dump(random_forest, f)

# 进行预测
y_pred = random_forest.predict(X_test)

# 绘制实际值与预测值的散点图
plt.scatter(y_test, y_pred)
x = np.linspace(0, 50, 50)
plt.plot(x, x)
plt.title('Actual vs Predicted (Random Forest)')
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.savefig('../figs/random_forest_regressor.png')
plt.show()

# 计算并打印MSE和R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

update_model_results('Random Forest', mse, r2)
