from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

import sys

import pre_processing as pp


def calculate_RMSE(y_pred, y_true):
    return mean_squared_error(y_true, y_pred) 

def MNB_test():
    x, y = pp.read_data()
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)

    clf = MultinomialNB()
    clf.fit(train_x, train_y)

    rst = clf.predict(test_x)
    print calculate_RMSE(rst, test_y)
    print pp.print_mean_median(rst)

def MLR_test():
    x, y = pp.read_data()
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=4)
    clf.fit(train_x, train_y)

    rst = clf.predict(test_x)

    print calculate_RMSE(rst, test_y)
    print pp.print_mean_median(rst)


def tree_test():
    x, y = pp.read_data()
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)

    rst = clf.predict(test_x)

    train_rst = clf.predict(train_x)

    print calculate_RMSE(train_rst, train_y)

    print calculate_RMSE(rst, test_y)
    print pp.print_mean_median(rst)


def random_forest_test():
    x, y = pp.read_data()
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)

    clf = RandomForestClassifier()

    clf.fit(train_x, train_y)
    rst = clf.predict(test_x)

    print calculate_RMSE(rst, test_y)
    print pp.print_mean_median(rst)

def main():
    # tree_test()

    random_forest_test()

if __name__ == '__main__':
    main()

