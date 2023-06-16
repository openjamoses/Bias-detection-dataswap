
from sklearn.metrics import accuracy_score, confusion_matrix
from copy import deepcopy
from src.common.metrics_utils import FairnessMetrics_binary
from src.common.utility_functions import *
import pandas as pd
import numpy as np


class FairWAY:
    def __init__(self, train, test, colums_list, target_index, acc_, dict_fairness_metrics):
        self.target_index = target_index
        self.acc = acc_
        self.train, self.test = train, test
        self.colums_list = colums_list
        self.dict_fairness_metrics = dict_fairness_metrics

        #print('before: ', self.test.shape, type(self.test))
        train_transformed = self.data_binning(self.train, 0)
        test_transformed = self.data_binning(self.test, 0)
        #train_transformed.reshape((self.train.shape))
        #test_transformed.reshape((self.test.shape))
        #print('after: ', test_transformed.shape, type(test_transformed))
        for column_id in range(1, self.train.shape[1]):
            train_transformed = self.data_binning(train_transformed, column_id)
            test_transformed = self.data_binning(test_transformed, column_id)
        self.train_transformed = train_transformed
        self.test_transformed = test_transformed
        self.x_train, self.y_train = split_features_target(train_transformed, self.target_index)
        self.x_test, self.y_test = split_features_target(test_transformed, self.target_index)
        #self.data_writer_baseline = data_writer_baseline



        y_list = [y for y in self.y_train]
        for y in self.y_test:
            y_list.append(y)
        unique = np.unique(y_list)
        self.label = y_list
        self.fav = unique[1] if unique[0] < unique[1] else unique[0]
        self.non_fav = unique[1] if unique[0] > unique[1] else unique[0]

    def data_binning(self, data, feature_index):
        # data_range = self.data[0:, feature_index]
        #print(type, data.shape)
        n, m = data.shape
        x_transform = np.zeros(shape=(n,m))
        data_range = data[0:, feature_index]
        folded_data = {}
        unique_ = np.unique(data_range)
        if len(unique_) == 2:
            folded_data[0.0] = [np.unique(data_range)[0] if np.unique(data_range)[0] < np.unique(data_range)[1] else
                                np.unique(data_range)[1]]
            folded_data[1.0] = [np.unique(data_range)[0] if np.unique(data_range)[0] > np.unique(data_range)[1] else \
                                    np.unique(data_range)[1]]
            x_transform = data
        elif len(unique_) > 2 and len(unique_) <= 6:
            medium = np.median(unique_)
            folded_data[0.0] = [0.0 for v in unique_ if v <= medium]
            folded_data[1.0] = [1.0 for v in unique_ if v > medium]
            for i in range(len(data_range)):
                x_transform[i,:] = data[i,:]
                if data[i,feature_index] <= medium:
                    x_transform[i, feature_index] = 0.0
                else:
                    x_transform[i, feature_index] = 1.0
        else:
            # print(data_range)
            # percentile_50_ = (min(data_range) + max(data_range))/2 #np.percentile(list(set(data_range)), 50)
            percentile_50_ = np.percentile(unique_, 50)
            # percentile_50_ = np.mean(data_range)
            percentile_75 = np.percentile(data_range, 75)
            percentile_50 = max([i for i in unique_ if i <= percentile_50_])
            percentile_100 = np.percentile(data_range, 100)
            # print('percentile_50: ', percentile_50, percentile_100, np.unique(data_range), data_range)
            if percentile_50 == np.min(data_range) or percentile_50 == np.max(data_range):
                # percentile_25 = np.percentile(np.unique(data_range), 25)
                percentile_50 = np.percentile(np.unique(data_range), 50)
                # percentile_75 = np.percentile(np.unique(data_range), 75)
            # if percentile_50 == percentile_25:
            #    percentile_50 = np.max(data_range)/2
            # if percentile_25 == np.min(data_range):
            #    percentile_25 = percentile_50/2
            for i in range(len(data_range)):
                fold_id = percentile_50
                x_transform[i, :] = data[i, :]
                if data_range[i] <= percentile_50:
                    fold_id = 0.0
                else:  # and data_range[i] <= percentile_50:
                    fold_id = 1.0
                x_transform[i, feature_index] = fold_id
        return x_transform
    def find_class_protected(self):
        protected_dict = {}
        for column_id in range(self.x_train.shape[1]):
            data_ = self.x_train[0:, column_id]
            c = np.unique(data_)
            c0 = np.sum(data_ == c[0])
            c1 = np.sum(data_ == c[1])
            protected_ = c[0]
            non_protected_ = c[1]
            if c1 > c0:
                protected_ = c[1]
                non_protected_ = c[0]
            protected_dict[column_id] = (protected_, non_protected_)
        return protected_dict

    def fit_baseline(self, clf, p_pred, fold, droped_attrib, data_writer_baseline, random_seed=42, split_size=0.25):

        data_test_fm_ = {}
        data_train_fm_ = {}
        # train_transformed = self.data_binning(self.train, 0)
        # test_transformed = self.data_binning(self.test, 0)

        data_combined_fm_ = {}
        for column_id in range(self.train.shape[1]):

            # train_transformed = self.data_binning(train_transformed, column_id)
            # test_transformed = self.data_binning(test_transformed, column_id)
            data_train_fm_[self.colums_list[column_id]] = self.train_transformed[0:, column_id]
            data_test_fm_[self.colums_list[column_id]] = self.test_transformed[0:, column_id]
            data_combined_fm_[self.colums_list[column_id]] = self.train_transformed[0:, column_id]
            data_combined_fm_[self.colums_list[column_id]] = self.test_transformed[0:, column_id]
        data_train_fm = pd.DataFrame(data_train_fm_)
        data_test_fm = pd.DataFrame(data_test_fm_)
        data_combined_fm = pd.DataFrame(data_combined_fm_)

        protected_dict = self.find_class_protected()

        clf_protected = deepcopy(clf)
        clf_non_protected = deepcopy(clf)
        clf_base = deepcopy(clf)

        for column_id in range(self.x_train.shape[1]):
            TPR_difference, FPR_difference, SPD, DIR, AOD, TPR, FPR, SP, Precision, Recal, F1, ACC = self.dict_fairness_metrics[column_id]

            (protected_, non_protected_) = protected_dict[column_id]

            # divide the data based on protected group
            dataset_orig_protected, dataset_orig_non_protected = [x for _, x in data_combined_fm.groupby(data_combined_fm[self.colums_list[column_id]] == protected_)]

            dataset_orig_protected[self.colums_list[column_id]] = protected_

            X_train_protected, y_train_protected = dataset_orig_protected.loc[:, dataset_orig_protected.columns != self.colums_list[self.target_index]], \
            dataset_orig_protected[self.colums_list[self.target_index]]


            clf_protected.fit(X_train_protected, y_train_protected)
            #y_protected = np.arange(len(dataset_orig_protected.columns) - 1)

            ## None protected
            X_train_non_protected, y_train_non_protected = dataset_orig_non_protected.loc[:, dataset_orig_non_protected.columns != self.colums_list[self.target_index]], \
            dataset_orig_non_protected[self.colums_list[self.target_index]]

            clf_non_protected.fit(X_train_non_protected, y_train_non_protected)
            #y_non_protected = np.arange(len(dataset_orig_non_protected.columns) - 1)

            ## Remove biased rows
            df_removed = pd.DataFrame(columns=data_combined_fm.columns)

            for index, row in data_combined_fm.iterrows():
                row_ = [row.values[0:len(row.values) - 1]]
                y_protected = clf_protected.predict(row_)
                y_non_protected = clf_non_protected.predict(row_)
                if y_protected[0] != y_non_protected[0]:
                    df_removed = df_removed.append(row, ignore_index=True)
                    data_combined_fm = data_combined_fm.drop(index)

            np.random.seed(random_seed)
            ## Divide into train,validation,test
            dataset_orig_train, dataset_orig_test = train_test_split(data_combined_fm, test_size=split_size, random_state=random_seed,
                                                                     shuffle=True, stratify=self.label)

            X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != self.colums_list[self.target_index]], \
            dataset_orig_train[self.colums_list[self.target_index]]
            X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != self.colums_list[self.target_index]], dataset_orig_test[
                self.colums_list[self.target_index]]

            y_test = y_test.tolist()

            clf_base.fit(X_train, y_train)
            y_pred = clf_base.predict(X_test)
            y_pred = y_pred.tolist()

            column_data = X_test[self.colums_list[column_id]].tolist()

            print('y_test: ', np.unique(y_test), 'y_pred: ', np.unique(y_pred), 'column_data: ', np.unique(column_data))

            fairnessMetrics_binary = FairnessMetrics_binary(column_data,protected_key=protected_,non_protected_key=non_protected_,fav=self.fav,
                                                            non_fav=self.non_fav)
            TPR_diff_after, FPR_diff_after, SPD_after, DIR_after, AOD_after, TPR_after, FPR_after, SP_after, Precision_after, Recal_after, F1_after, ACC_after = fairnessMetrics_binary.fairness_metrics(y_test, y_pred)
            acc_after = accuracy_score(y_pred, y_test)

            data_writer_baseline.writerow(
                [fold, self.colums_list[column_id], droped_attrib, self.acc, acc_after, TPR_difference, TPR_diff_after, FPR_difference, FPR_diff_after,
                 SPD, SPD_after, DIR,
                 DIR_after, AOD, AOD_after, TPR, TPR_after, FPR, FPR_after, SP, SP_after, Precision,
                 Precision_after, Recal, Recal_after, F1, F1_after, ACC, ACC_after])