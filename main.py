import pandas as pd
import numpy as np
from KNN import KNN
from sklearn.model_selection import train_test_split


def get_accuracy(y_true, y_predicted):
    accuracy = np.sum(y_true == y_predicted) / len(y_true)
    return accuracy


dataset = pd.read_csv(
    "C:\\Users\\aliam\\PycharmProjects\\Assginment_1_Supervised_Learning\\MNIST_Train_Preprocessed_1.csv",
    header=0,
    index_col=0)
test_set = pd.read_csv(
    "C:\\Users\\aliam\\PycharmProjects\\Assginment_1_Supervised_Learning\\MNIST_Test_Preprocessed_1.csv",
    header=0,
    index_col=0)

y = dataset["label"]
X = dataset.drop("label", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 3
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("KNN accuracy: " + str(get_accuracy(y_test, predictions)))
