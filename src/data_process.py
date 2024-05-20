import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def process_data():
    boston = pd.read_csv('../dataset/boston.csv')

    X = boston.drop(['MEDV'], axis=1)   # Feature data
    y = boston['MEDV']                        # Target data
    # Split train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    pickle.dump(scaler, open('../models/scaler.pkl', 'wb'))

    # PCA
    pca = PCA(n_components=0.90)     # Keep the components that explain 99% of the variance
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    pickle.dump(pca, open('../models/pca.pkl', 'wb'))
    
    # print the most relevant features
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Components:\n", pca.components_)


    # Save processed data
    with open('../dataset/processed_data.pkl', 'wb') as f:
        pickle.dump((X_train_pca, X_test_pca, y_train, y_test), f)


if __name__ == '__main__':
    process_data()
