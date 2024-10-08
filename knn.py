# -*- coding: utf-8 -*-
"""kNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1s2dGojUCzJVMR7C4ta42zNrbn0Hyj9_J

## scikit-learn Approach
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
data, labels = iris.data, iris.target

train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                       train_size=0.8,
                       test_size=0.2,
                       random_state=12)

from sklearn.neighbors import KNeighborsClassifier
# classifier "out of the box", no parameters
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)

print("Predictions from the classifier:")
test_data_predicted = knn.predict(test_data)
print(test_data_predicted)
print("Target values:")
print(test_labels)

print(accuracy_score(test_data_predicted, test_labels))

"""## scikit-learn Approach (Hypperparameter Tuning)"""

knn = KNeighborsClassifier(algorithm='auto',
                           leaf_size=30,
                           metric='minkowski',
                           p=2,
                           metric_params=None,
                           n_jobs=1,
                           n_neighbors=5,
                           weights='uniform')

"""## From Scratch"""

