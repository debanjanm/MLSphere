import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from base_data_operations import create_classification_dataset

class BaseML:
    def __init__(self, model):
        self.model = model
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        X, y = create_classification_dataset()
        self.X = X.values
        self.y = y.values
        print("Data loaded successfully.")
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")

    def split_data(self, test_size=0.2, random_state=42):
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print("Data split successfully.")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate_model(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split. Call split_data() first.")
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

    def predict_sample(self, sample_index=0):
        if self.X_test is None:
            raise ValueError("Data not split. Call split_data() first.")
        sample = self.X_test[sample_index].reshape(1, -1)
        prediction = self.model.predict(sample)
        print(f"\nPrediction for sample at index {sample_index}: {prediction[0]}")
        print(f"Actual value: {self.y_test[sample_index]}")