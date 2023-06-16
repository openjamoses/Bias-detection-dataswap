
from sklearn.metrics import accuracy_score, confusion_matrix
from aif360.datasets import StandardDataset
from sklearn.model_selection import GridSearchCV
from src.common.metrics_utils import FairnessMetrics_binary
from src.common.utility_functions import *
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from aif360.algorithms.preprocessing import DisparateImpactRemover

class AIF360Disperate_Remover:
    def __init__(self, train, test, colums_list, target_index, acc_, dict_fairness_metrics):
        self.target_index = target_index
        self.acc = acc_
        self.train, self.test = train, test
        self.colums_list = colums_list
        self.colums_list = [str(c).strip() for c in self.colums_list]
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

    def fit_baseline(self, clf, p_pred, fold, droped_attrib, data_writer_baseline):
        data_test_fm_ = {}
        data_train_fm_ = {}
        # train_transformed = self.data_binning(self.train, 0)
        # test_transformed = self.data_binning(self.test, 0)
        for column_id in range(self.train.shape[1]):
            # train_transformed = self.data_binning(train_transformed, column_id)
            # test_transformed = self.data_binning(test_transformed, column_id)
            data_train_fm_[self.colums_list[column_id]] = self.train_transformed[0:, column_id]
            data_test_fm_[self.colums_list[column_id]] = self.test_transformed[0:, column_id]
        data_train_fm = pd.DataFrame(data_train_fm_, columns=self.colums_list)
        data_test_fm = pd.DataFrame(data_test_fm_, columns=self.colums_list)

        x_train, y_train = data_train_fm.loc[:,
                                       data_train_fm.columns != self.colums_list[self.target_index]], data_train_fm[
            self.colums_list[self.target_index]]
        x_test, y_test = data_test_fm.loc[:, data_test_fm.columns != self.colums_list[self.target_index]], \
            data_test_fm[self.colums_list[self.target_index]]

        data_train_fm_copy1 = data_train_fm.copy()
        data_train_fm_copy2 = data_train_fm.copy()

        data_test_fm_copy1 = data_test_fm.copy()
        data_test_fm_copy2 = data_test_fm.copy()
        protected_dict = self.find_class_protected()

        clf_base = deepcopy(clf)
        for column_id in range(self.x_train.shape[1]):
            TPR_difference, FPR_difference, SPD, DIR, AOD, TPR, FPR, SP, Precision, Recal, F1, ACC = self.dict_fairness_metrics[column_id]

            (protected_, non_protected_) = protected_dict[column_id]

            privileged_groups = [{self.colums_list[column_id]: non_protected_}]
            unprivileged_groups = [{self.colums_list[column_id]: protected_}]
            train_std2 = StandardDataset(data_train_fm_copy2,
                                         label_name=self.colums_list[self.target_index],
                                         favorable_classes=[self.fav],
                                         protected_attribute_names=[self.colums_list[column_id]],
                                         privileged_classes=[[protected_], [non_protected_]])

            test_std2 = StandardDataset(data_test_fm_copy2,
                                        label_name=self.colums_list[self.target_index],
                                        favorable_classes=[self.fav],
                                        protected_attribute_names=[self.colums_list[column_id]],
                                        privileged_classes=[[protected_], [non_protected_]])

            index = train_std2.feature_names.index(self.colums_list[column_id])
            for level in tqdm(np.linspace(0., 1., 11)):
                di = DisparateImpactRemover(repair_level=level)
                train_repd = di.fit_transform(train_std2)
                test_repd = di.fit_transform(test_std2)

                column_data = test_repd.features[0:, index]

                X_tr = np.delete(train_repd.features, index, axis=1)
                X_te = np.delete(test_repd.features, index, axis=1)
                y_tr = train_repd.labels.ravel()

                clf_base.fit(X_tr, y_tr)

                test_repd_pred = test_repd.copy()
                y_pred = clf_base.predict(X_te)

                fairnessMetrics_binary = FairnessMetrics_binary(column_data, protected_key=protected_,
                                                                non_protected_key=non_protected_, fav=self.fav,
                                                                non_fav=self.non_fav)
                TPR_diff_after, FPR_diff_after, SPD_after, DIR_after, AOD_after, TPR_after, FPR_after, SP_after, Precision_after, Recal_after, F1_after, ACC_after = fairnessMetrics_binary.fairness_metrics(
                    test_repd_pred.labels, y_pred)
                acc_after = accuracy_score(y_pred, test_repd_pred.labels)

                data_writer_baseline.writerow(
                    [fold, level, self.colums_list[column_id], droped_attrib, self.acc, acc_after, TPR_difference, TPR_diff_after,
                     FPR_difference, FPR_diff_after,
                     SPD, SPD_after, DIR,
                     DIR_after, AOD, AOD_after, TPR, TPR_after, FPR, FPR_after, SP, SP_after, Precision,
                     Precision_after, Recal, Recal_after, F1, F1_after, ACC, ACC_after])