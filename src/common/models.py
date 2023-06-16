import sklearn
import xgboost
import sklearn
#from tensorflow.keras.utils.np_utils import to_categorical
from sklearn import naive_bayes, neighbors, tree, linear_model, ensemble
from sklearn.naive_bayes import GaussianNB
from tensorflow.python.keras.utils.np_utils import to_categorical
from src.common.NNClassifier import NNClassifier
#from src.models.v3.NNClassifier import NNClassifier
from src.common.utility_functions import *


class Models:
    def __init__(self, train, test, target_index, normalize=False, nn=False):
        self.x_train, self.y_train = split_features_target(train, index=target_index)
        self.x_test, self.y_test = split_features_target(test, index=target_index)
        #if nn:

        self.normalize = normalize
        if self.normalize:
            self.normalise_data()
    def normalise_data(self):
        # normalize data
        self.x_train_norm = (self.x_train - self.x_train.mean(axis=0)) / self.x_train.std(axis=0)
        self.x_train_norm.mean(axis=0)

        self.x_test_norm = (self.x_test - self.x_test.mean(axis=0)) / self.x_test.std(axis=0)
        self.x_test_norm.mean(axis=0)
    def gaussianNB(self):
        if self.normalize:
            self.model = naive_bayes.GaussianNB().fit(self.x_train_norm, self.y_train)
            #self.model = xgboost.XGBClassifier().fit(self.x_train_norm, self.y_train)
        else:
            self.model = naive_bayes.GaussianNB().fit(self.x_train, self.y_train)
            #self.model = xgboost.XGBClassifier().fit(self.x_train, self.y_train)

        return self.model
    def xgboost_regressor(self):
        if self.normalize:
            self.model = xgboost.XGBClassifier().fit(self.x_train_norm, self.y_train)
            #self.model = xgboost.XGBClassifier().fit(self.x_train_norm, self.y_train)
        else:
            self.model = xgboost.XGBClassifier().fit(self.x_train, self.y_train)
            #self.model = xgboost.XGBClassifier().fit(self.x_train, self.y_train)
        return self.model
    def knn_regressor(self):
        if self.normalize:
            self.model = neighbors.KNeighborsRegressor().fit(self.x_train_norm, self.y_train)
        else:
            self.model = neighbors.KNeighborsRegressor().fit(self.x_train, self.y_train)
        return self.model
    def decision_tree_regressor(self):
        if self.normalize:
            self.model = tree.DecisionTreeClassifier().fit(self.x_train_norm, self.y_train)

        else:
            self.model = tree.DecisionTreeClassifier().fit(self.x_train, self.y_train)
        return self.model
    def logistic_regression(self):
        if self.normalize:
            self.model = linear_model.LogisticRegression().fit(self.x_train_norm, self.y_train)
        else:
            self.model = linear_model.LogisticRegression().fit(self.x_train, self.y_train)
        return self.model
    def random_forest(self):
        if self.normalize:
            self.model = ensemble.RandomForestRegressor().fit(self.x_train_norm, self.y_train)
        else:
            self.model = ensemble.RandomForestRegressor().fit(self.x_train, self.y_train)
        return self.model
    def nn_classifier(self):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        if self.normalize:
            self.model = NNClassifier(self.x_train.shape[1], n_classes)
            self.model.fit(self.x_train_norm, y_train)
        else:
            self.model = NNClassifier(self.x_train.shape[1], n_classes)
            self.model.fit(self.x_train, y_train)
        return self.model
    def model_predict(self):
        if self.normalize:
            pred = self.model.predict(self.x_test_norm)
        else:
            pred = self.model.predict(self.x_test)
        return pred
    def accuracy(self, y_test, pred):
        print('Before: ', np.array(y_test))
        print('After: ', np.array(pred))
        correct = np.sum(np.array(y_test) == np.array(pred))
        print('Accuracy: ', correct*100/y_test.shape[0])
        print('Accuracy Score: ', sklearn.metrics.accuracy_score(np.array(y_test), np.array(pred)))
        print('Precision Score: ', sklearn.metrics.precision_score(np.array(y_test), np.array(pred)))
        print('Recal Score: ', sklearn.metrics.recall_score(np.array(y_test), np.array(pred)))
        print('F1 Score: ', sklearn.metrics.f1_score(np.array(y_test), np.array(pred)))





class Models2:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train, self.y_train = x_train, y_train #  split_features_target(train, index=target_index)
        self.x_test, self.y_test =  x_test, y_test # split_features_target(test, index=target_index)
        #if nn:

        #self.normalize = normalize
        #if self.normalize:
        #    self.normalise_data()
    def normalise_data(self):
        # normalize data
        self.x_train_norm = (self.x_train - self.x_train.mean(axis=0)) / self.x_train.std(axis=0)
        self.x_train_norm.mean(axis=0)

        self.x_test_norm = (self.x_test - self.x_test.mean(axis=0)) / self.x_test.std(axis=0)
        self.x_test_norm.mean(axis=0)
    def gaussianNB(self):
        self.model = GaussianNB().fit(self.x_train, self.y_train)
        return self.model
    def xgboost_regressor(self):
        self.model = xgboost.XGBClassifier().fit(self.x_train, self.y_train)
        return self.model
    def knn_regressor(self):
        self.model = sklearn.neighbors.KNeighborsRegressor().fit(self.x_train, self.y_train)
        return self.model
    def decision_tree_regressor(self):
        self.model = sklearn.tree.DecisionTreeRegressor().fit(self.x_train, self.y_train)
        return self.model
    def logistic_regression(self):
        self.model = sklearn.linear_model.LogisticRegression().fit(self.x_train, self.y_train)
        return self.model
    def random_forest(self):
        self.model = sklearn.ensemble.RandomForestRegressor().fit(self.x_train, self.y_train)
        return self.model
    def nn_classifier(self):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)
        n_classes = y_test.shape[1]
        self.model = NNClassifier(self.x_train.shape[1], n_classes)
        self.model.fit(self.x_train, y_train)
        return self.model
    def model_predict(self):
        pred = self.model.predict(self.x_test)
        return pred
    def accuracy(self, y_test, pred):
        print('Before: ', np.array(y_test))
        print('After: ', np.array(pred))
        correct = np.sum(np.array(y_test) == np.array(pred))
        print('Accuracy: ', correct*100/y_test.shape[0])
        print('Accuracy Score: ', sklearn.metrics.accuracy_score(np.array(y_test), np.array(pred)))
        print('Precision Score: ', sklearn.metrics.precision_score(np.array(y_test), np.array(pred)))
        print('Recal Score: ', sklearn.metrics.recall_score(np.array(y_test), np.array(pred)))
        print('F1 Score: ', sklearn.metrics.f1_score(np.array(y_test), np.array(pred)))

