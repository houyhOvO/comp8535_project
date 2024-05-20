import pickle
import numpy as np

from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

with open('../dataset/processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Make sure the data is writable
X_train = np.copy(X_train)
y_train = np.copy(y_train)

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)

# save the model
with open('../models/rbf_svr.pkl', 'wb') as f:
    pickle.dump(rbf_svr, f)

y_pred = rbf_svr.predict(X_test)

plt.scatter(y_test,y_pred)
x = np.linspace(0, 50, 50)
plt.plot(x, x)
plt.title('Actual vs Predicted (Support Vector Machine )')
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.savefig('../figs/support_vector_machine.png')
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)
