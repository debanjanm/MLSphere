import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def main():
    # This main function is just for demonstration
    # You would typically import this class and use it in other files
    from base_ml_operations import BaseML

    # Create a model (using sklearn's KNN as an example)
    knn_model = KNN(k=5)

    # Create an instance of BaseML with the model
    ml_ops = BaseML(knn_model)

    # Demonstrate the usage
    ml_ops.load_data()
    ml_ops.split_data()
    ml_ops.train_model()
    ml_ops.evaluate_model()
    ml_ops.predict_sample()

if __name__ == "__main__":
    main()