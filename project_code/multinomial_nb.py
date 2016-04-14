from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

import sys

import pre_processing as pp



def calculate_RMSE(y_pred, y_true):
    return mean_squared_error(y_true, y_pred)**0.5

def crossvalidation(clf, X, Y):
    scores = cross_validation.cross_val_score(clf, X, Y, cv=5, scoring='mean_squared_error')
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def MNB_test(x, y):
    print 'MNB test'
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)

    clf = MultinomialNB()
    crossvalidation(clf, x, y)


    clf.fit(train_x, train_y)
    rst = clf.predict(test_x)
    print calculate_RMSE(rst, test_y)
    print pp.print_mean_median(rst)

def MLR_test(x, y):
    print 'MLR test'    
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=4)
    crossvalidation(clf, x, y)



    clf.fit(train_x, train_y)
    rst = clf.predict(test_x)
    print calculate_RMSE(rst, test_y)
    print pp.print_mean_median(rst)


def tree_test(x, y):
    print 'DecisionTree'
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)
    clf = tree.DecisionTreeClassifier()
    crossvalidation(clf, x, y)


    clf.fit(train_x, train_y)
    rst = clf.predict(test_x)

    train_rst = clf.predict(train_x)

    print calculate_RMSE(train_rst, train_y)

    print calculate_RMSE(rst, test_y)
    print pp.print_mean_median(rst)


def svm_test(x, y):
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)

    clf = svm.SVC(C=0.001, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    print 'fitting: '
    crossvalidation(clf, x, y)


    clf.fit(train_x, train_y)

    print 'predicting: '
    rst = clf.predict(test_x)
    # print sum(rst == test_y) / len(test_y) * 1.0
    print calculate_RMSE(rst, test_y)   
    pp.print_mean_median(rst)



def random_forest_test(x, y):
    print 'RandomForest', 
    train_x, train_y, test_x, test_y = pp.split_data_set(x, y)

    clf = RandomForestClassifier()

    crossvalidation(clf, x, y)

    clf.fit(train_x, train_y)
    rst = clf.predict(test_x)

    print rst
    print test_y
    print calculate_RMSE(rst, test_y)
    print 'mean rst: ', 
    pp.print_mean_median(rst)

def main():
    # tree_test(x, y)
    if len(sys.argv) > 0:
        x, y = pp.read_data(dir_path = sys.argv[1])
    else:
        x, y = pp.read_data()

    func_list = [MLR_test, tree_test, random_forest_test]
    # random_forest_test(x, y)
    # MNB_test(x, y)

    for func in func_list:
        func(x, y)


    print 'end_main'

if __name__ == '__main__':
    main()

