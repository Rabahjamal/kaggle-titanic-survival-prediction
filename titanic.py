from data_reader.reader import CsvReader
from util import *
import numpy as np
import matplotlib.pyplot as plt
import csv


class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, epochs=50):
        self.__epochs= epochs
        self.__learning_rate = learning_rate

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.__epochs):
            y_ = self.__net_input(X)
            activated_y = self.__activation(y_)
            errors = (y - activated_y)
            neg_grad = X.T.dot(errors)
            self.w_[1:] += self.__learning_rate * neg_grad
            self.w_[0] += self.__learning_rate * errors.sum()

            self.cost_.append(self.__logit_cost(y, self.__activation(y_)))

    def __logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))

        return logit

    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def __net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def __activation(self, X):
        return self.__sigmoid(X)

    def predict(self, X):
        z = self.__net_input(X)
        return np.where(self.__activation(z) >= 0.5, 1, 0)

#reading train dataset
reader = CsvReader("./data/titanic/train.csv")
titanic_features, titanic_labels = reader.get_titanic_train_data()

#reading test(features) set
reader = CsvReader("./data/titanic/test.csv")
titanic_test_features = reader.get_titanic_test_data()


titanic_features, titanic_labels = shuffle(titanic_features, titanic_labels)

train_x, train_y, test_x = titanic_features[0:892], titanic_labels[0:892], titanic_test_features[0:419]
train_x, train_y, test_x = np.asarray(train_x), np.asarray(train_y), np.asarray(test_x)


train_x, means, stds = standardize(train_x)
test_x = standardize(test_x, means, stds)


lr = LogisticRegression(learning_rate=0.001, epochs=50)
lr.fit(train_x, train_y)

plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Titanic - Learning rate 0.001')

plt.tight_layout()
plt.show()

predicted_test = lr.predict(test_x)

#creating csv file
data = [['PassengerId', 'Survived']]
id = 892

for item in predicted_test[0:]:
    data.append(list([id, item]))
    id = id + 1

with open('data/titanic/finalResult.csv', 'w', newline='') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)


print(predicted_test)
