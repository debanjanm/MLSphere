def main():
    # This main function is just for demonstration
    # You would typically import this class and use it in other files
    from base_ml_operations import BaseML
    from sklearn.neighbors import KNeighborsClassifier

    # Create a model (using sklearn's KNN as an example)
    knn_model = KNeighborsClassifier(n_neighbors=5)

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