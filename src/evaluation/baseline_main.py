import os

import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression
from src.baseline.aif360_disperate_remover import AIF360Disperate_Remover
from src.baseline.aif360_reweighing import AIF360Reweigh
from src.baseline.fairsmote import FairSMOTE
from src.common.metrics_utils import FairnessMetrics_partioning
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from src.common.load_data import LoadData
from src.common.utility_functions import *

from src.baseline.LRTD import LRTD


class Sensitive:
    def __init__(self, df_data, target_index, target_name, data_name, colums_list, log_path, threshold=1):
        self.df_data = df_data
        self.target_index = target_index
        self.target_name = target_name
        self.data_name = data_name
        self.colums_list = colums_list
        self.log_path = log_path
        self.threshold = threshold
        self.init_csv()
        self.init_data()
    def init_data(self, drop=[]):
        if len(drop) > 0:
            self.df_data_copy = self.df_data.copy()
            self.df_data_copy = self.df_data_copy.drop(drop, axis=1)
            self.colums_list = self.df_data_copy.columns.tolist()
            self.target_index = self.colums_list.index(self.target_name)
            self.data = self.df_data_copy.to_numpy()
        else:
            #self.df_data_copy = self.df_data.copy()
            #self.df_data_copy = self.df_data_copy.drop(drop, axis=1)
            self.colums_list = self.df_data.columns.tolist()
            self.target_index = self.colums_list.index(self.target_name)
            self.data = self.df_data.to_numpy()
    def init_csv(self):
        self.log_data = self.data_name
        self.new_path = self.log_path
        if self.threshold == None:
            id = '-baseline'
        else:
            id = self.threshold
        # if self.log_data:
        if not os.path.exists(self.log_path + str(self.data_name)):
            os.makedirs(self.log_path + str(self.data_name))
        if not os.path.exists(self.log_path + self.data_name + "/{}".format(id)):
            os.makedirs(self.log_path + str(self.data_name) + "/{}".format(id))
        self.new_path = self.log_path + str(self.data_name) + "/{}/".format(id)

        self.log_path_baseline = self.new_path + 'Baseline_LRTD_{}.csv'.format(self.data_name)
        self.data_file_baseline = open(self.log_path_baseline, mode='w', newline='',
                                       encoding='utf-8')
        self.data_writer_baseline = csv.writer(self.data_file_baseline)
        self.data_writer_baseline.writerow(
            ['Fold', 'Feature', 'Droped_Attrib','acc_','acc_after', 'TPR_diff','TPR_diff_after', 'FPR_diff', 'FPR_diff_after', 'SPD', 'SPD_after', 'DIR',
             'DIR_after', 'AOD', 'AOD_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after', 'SP', 'SP_after', 'Pre',
             'Pre_after', 'Re', 'Re_after', 'F1', 'F1_after', 'ACC', 'ACC_after'])

        self.log_path_baseline_smote = self.new_path + 'Baseline_fairSMOTE_{}.csv'.format(self.data_name)
        self.data_file_baseline_fairsmote = open(self.log_path_baseline_smote, mode='w', newline='',
                                       encoding='utf-8')
        self.data_writer_baseline_fairsmote = csv.writer(self.data_file_baseline_fairsmote)
        self.data_writer_baseline_fairsmote.writerow(
            ['Fold', 'Feature','Droped_Attrib', 'acc_', 'acc_after', 'TPR_diff', 'TPR_diff_after', 'FPR_diff', 'FPR_diff_after', 'SPD',
             'SPD_after', 'DIR',
             'DIR_after', 'AOD', 'AOD_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after', 'SP', 'SP_after', 'Pre',
             'Pre_after', 'Re', 'Re_after', 'F1', 'F1_after', 'ACC', 'ACC_after'])

        self.data_file_baseline_reweighing = open(self.new_path + 'Baseline_Reweighing_{}.csv'.format(self.data_name), mode='w', newline='',  encoding='utf-8')
        self.data_writer_baseline_reweighing = csv.writer(self.data_file_baseline_reweighing)
        self.data_writer_baseline_reweighing.writerow(
            ['Fold', 'Feature', 'Droped_Attrib', 'acc_', 'acc_after', 'TPR_diff', 'TPR_diff_after', 'FPR_diff',
             'FPR_diff_after', 'SPD',
             'SPD_after', 'DIR',
             'DIR_after', 'AOD', 'AOD_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after', 'SP', 'SP_after', 'Pre',
             'Pre_after', 'Re', 'Re_after', 'F1', 'F1_after', 'ACC', 'ACC_after'])

        self.data_file_baseline_di_remover = open(self.new_path + 'Baseline_DIR_{}.csv'.format(self.data_name),
                                                  mode='w', newline='', encoding='utf-8')
        self.data_writer_baseline_di_remover = csv.writer(self.data_file_baseline_di_remover)
        self.data_writer_baseline_di_remover.writerow(
            ['Fold', 'Level','Feature', 'Droped_Attrib', 'acc_', 'acc_after', 'TPR_diff', 'TPR_diff_after', 'FPR_diff',
             'FPR_diff_after', 'SPD',
             'SPD_after', 'DIR',
             'DIR_after', 'AOD', 'AOD_after', 'TPR', 'TPR_after', 'FPR', 'FPR_after', 'SP', 'SP_after', 'Pre',
             'Pre_after', 'Re', 'Re_after', 'F1', 'F1_after', 'ACC', 'ACC_after'])

    def _random_select_indices(self, y, proportion_):
        y_indices = [i for i in range(len(y))]
        N = int(round(len(y)*proportion_, 0))
        #print(y_indices, N)
        return random.sample(y_indices, k=N)  # Four samples without replacement

    def __get_protected_attrib(self, column_id):
        cd = data_partioning(self.data, column_id)
        # print('size:', len(cd.keys()), 'category_data: ', cd)
        data_ = self.data[0:, column_id]
        data_sum = {}
        protected_dict = {}
        max_val = 0
        protected_1 = list(cd.keys())[0]
        for k, v in cd.items():
            sum_k = 0
            for v_ in v:
                sum_k += np.sum(data_ == v_)
            if sum_k > max_val:
                max_val = sum_k
                protected_1 = k
            data_sum[k] = sum_k

        # protected_dict[column_id] = protected_1
        non_protected_1 = list(cd.keys())[0]
        for p in cd.keys():
            if p != protected_1:
                non_protected_1 = p
                break
        return (protected_1, non_protected_1)
    def prob_ordering_freq(self, column_id, column_id2, data_partioning_dict):
        # feat2 = colums_list[column_id2]
        cd2 = data_partioning_dict[column_id2]
        cd = data_partioning_dict[column_id]

        P_a_b = 0
        P_a_not_b = 0
        P_b_not_a = 0
        TOTAL = len(self.data)
        # print(cd, cd2)
        x_1 = self.data[0:, column_id]
        x_2 = self.data[0:, column_id2]
        (protected_1, non_protected_1) = self.__get_protected_attrib(column_id)
        (protected_2, non_protected_2) = self.__get_protected_attrib(column_id2)
        for a in range(len(x_1)):
            # print('self.x_test[a]: ',self.x_test[a], self.x_test[column_id][a], self.x_test[a][column_id])
            if protected_1 in cd.keys() and protected_2 in cd2.keys():
                if x_1[a] in cd[protected_1] and x_2[a] in cd2[protected_2]:
                    P_a_b += 1
            if protected_1 in cd.keys() and non_protected_2 in cd2.keys():
                if x_1[a] in cd[protected_1] and x_2[a] in cd2[non_protected_2]:
                    P_a_not_b += 1
            if protected_2 in cd2.keys() and non_protected_1 in cd.keys():
                if x_2[a] in cd2[protected_2] and x_1[a] in cd[non_protected_1]:
                    P_b_not_a += 1
        # P_a_b_cond = np.count_nonzero((self.x_test == cd[protected_1]) & (self.x_test  2 == 1))
        P_a_b = P_a_b / TOTAL
        P_a_not_b = P_a_not_b / TOTAL
        P_b_not_a = P_b_not_a / TOTAL
        P_a_b_over_P_a_not_b = 0
        if P_a_not_b > 0:
            P_a_b_over_P_a_not_b = P_a_b / P_a_not_b
        P_b_a_over_P_b_not_a = 0
        if P_b_not_a > 0:
            P_b_a_over_P_b_not_a = P_a_b / P_b_not_a
        return (P_a_b, P_a_not_b, P_b_not_a, P_a_b_over_P_a_not_b, P_b_a_over_P_b_not_a)

    def fav_non_fav(self):
        y_list = [y for y in self.y_train]
        for y in self.y_test:
            y_list.append(y)
        unique = np.unique(y_list)
        self.fav = unique[1] if unique[0] < unique[1] else unique[0]
        self.non_fav = unique[1] if unique[0] > unique[1] else unique[0]
    def run_baseline_loop(self, train, test, drop=[]):
        if len(drop) > 0:
            self.init_data(drop=drop)
            train = train.drop(drop, axis=1)
            test = test.drop(drop, axis=1)
        data_partioning_dict = {}
        X, Y = split_features_target(self.data, index=self.target_index)
        for column_id in range(X.shape[1]):
            data_partioning_dict[column_id] = data_partioning(self.data, column_id)
        train, test = train.to_numpy(), test.to_numpy()
        self.x_train, self.y_train = split_features_target(train, self.target_index)
        self.x_test, self.y_test = split_features_target(test, self.target_index)

        self.fav_non_fav()

        ##TODO: We use LR coz we want to compare with the baseline in LRTD paper
        # y_train = to_categorical(self.y_train)
        # y_test = to_categorical(self.y_test)
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        NNmodel = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf.fit(self.x_train, self.y_train)
        NNmodel.fit(self.x_train, self.y_train)
        p_pred = NNmodel.predict(self.x_test)

        dict_fairness_metrics = {}
        for column_id in range(self.x_test.shape[1]):
            column_data = get_features_by_index(self.x_test, column_id)
            (protected_1, non_protected_1) = self.__get_protected_attrib(column_id)
            fairnessMetrics_partioning = FairnessMetrics_partioning(column_data, data_partioning_dict[column_id],
                                                                    protected_1, non_protected_1, self.fav,
                                                                    self.non_fav)
            dict_fairness_metrics[column_id] = fairnessMetrics_partioning.fairness_metrics(self.y_test, p_pred)
        return clf, p_pred, train, test, dict_fairness_metrics
    def fit(self, drop = [], repeat=10):
        X, Y = split_features_target(self.data, index=self.target_index)
        for _ in range(repeat):
            print(' ---- This is evaluation is being repeated now: ', _+1, ' times')
            kf = model_selection.KFold(n_splits=10)
            for fold, (train_idx, test_idx) in enumerate(kf.split(X=self.df_data,y=Y)):
                train, test = self.df_data.loc[train_idx], self.df_data.loc[test_idx]

                clf, p_pred, train_copy, test_copy, dict_fairness_metrics = self.run_baseline_loop(train, test)
                print(fold, train_copy.shape, test_copy.shape)
                print('  ---  Before dropping feature ')
                ## Call and run baseline here.
                ## TODO: LRTD
                Droped_Attrib = 'None'
                acc_ = accuracy_score(p_pred, self.y_test)
                train_copy_lrtdd, test_copy_lrtdd = train_copy.copy(), test_copy.copy()
                train_copy_fsmote, test_copy_fsmote = train_copy.copy(), test_copy.copy()
                train_copy_reweigh, test_copy_reweigh = train_copy.copy(), test_copy.copy()
                train_copy_dir, test_copy_dir = train_copy.copy(), test_copy.copy()
                baseline_lrtd = LRTD(train_copy_lrtdd, test_copy_lrtdd, self.colums_list, self.target_index, acc_, dict_fairness_metrics)
                baseline_lrtd.fit_baseline(clf, p_pred, fold, Droped_Attrib, self.data_writer_baseline)
                ## TODO: FairSMOTE
                baseline_fairsmote = FairSMOTE(train_copy_fsmote, test_copy_fsmote, self.colums_list, self.target_index, acc_, dict_fairness_metrics)
                baseline_fairsmote.fit_baseline(clf, p_pred, fold, Droped_Attrib, self.data_writer_baseline_fairsmote)
                # ## TODO: Reweighing
                baseline_reweighing = AIF360Reweigh(train_copy_reweigh, test_copy_reweigh, self.colums_list, self.target_index, acc_, dict_fairness_metrics)
                baseline_reweighing.fit_baseline(clf, p_pred, fold, Droped_Attrib, self.data_writer_baseline_reweighing)

                # ## TODO: DI Remover
                baseline_di_remover = AIF360Disperate_Remover(train_copy_dir, test_copy_dir, self.colums_list, self.target_index, acc_,dict_fairness_metrics)
                baseline_di_remover.fit_baseline(clf, p_pred, fold, Droped_Attrib, self.data_writer_baseline_di_remover)

                for drop_attrib in drop:
                    #Droped_Attrib = drop_attrib
                    ## todo: Lets rerun the test here after dropping the biased features

                    clf, p_pred, train_copy, test_copy, dict_fairness_metrics = self.run_baseline_loop(train, test, drop=drop_attrib)
                    print(fold, train_copy.shape, test_copy.shape)
                    print('  --- After droping the feature: ', drop_attrib)

                    train_copy_lrtdd, test_copy_lrtdd = train_copy.copy(), test_copy.copy()
                    train_copy_fsmote, test_copy_fsmote = train_copy.copy(), test_copy.copy()
                    train_copy_reweigh, test_copy_reweigh = train_copy.copy(), test_copy.copy()
                    train_copy_dir, test_copy_dir = train_copy.copy(), test_copy.copy()


                    acc_ = accuracy_score(p_pred, self.y_test)
                    baseline_lrtd = LRTD(train_copy_lrtdd, test_copy_lrtdd, self.colums_list, self.target_index, acc_, dict_fairness_metrics)
                    baseline_lrtd.fit_baseline(clf, p_pred, fold, drop_attrib, self.data_writer_baseline)
                    ## TODO: FairSMOTE


                    baseline_fairsmote = FairSMOTE(train_copy_fsmote, test_copy_fsmote, self.colums_list, self.target_index, acc_, dict_fairness_metrics)
                    baseline_fairsmote.fit_baseline(clf, p_pred, fold, drop_attrib, self.data_writer_baseline_fairsmote)
                    # ## TODO: Reweighing
                    baseline_reweighing = AIF360Reweigh(train_copy_reweigh, test_copy_reweigh, self.colums_list, self.target_index,
                                                        acc_, dict_fairness_metrics)
                    baseline_reweighing.fit_baseline(clf, p_pred, fold, drop_attrib,
                                                     self.data_writer_baseline_reweighing)

                    # ## TODO: DI Remover
                    baseline_di_remover = AIF360Disperate_Remover(train_copy_dir, test_copy_dir, self.colums_list,
                                                                  self.target_index, acc_, dict_fairness_metrics)
                    baseline_di_remover.fit_baseline(clf, p_pred, fold, drop_attrib,
                                                     self.data_writer_baseline_di_remover)

                self.init_data()

            #todo: close csv here
            self.data_file_baseline.close()
            self.data_file_baseline_fairsmote.close()
            self.data_file_baseline_reweighing.close()
            self.data_file_baseline_di_remover.close()

            self.log_path_baseline = self.new_path + 'Baseline_LRTD_{}.csv'.format(self.data_name)
            self.data_file_baseline = open(self.log_path_baseline, mode='a+', newline='',
                                           encoding='utf-8')
            self.data_writer_baseline = csv.writer(self.data_file_baseline)

            self.log_path_baseline_smote = self.new_path + 'Baseline_fairSMOTE_{}.csv'.format(self.data_name)
            self.data_file_baseline_fairsmote = open(self.log_path_baseline_smote, mode='a+', newline='',
                                                     encoding='utf-8')
            self.data_writer_baseline_fairsmote = csv.writer(self.data_file_baseline_fairsmote)

            self.data_file_baseline_reweighing = open(self.new_path + 'Baseline_Reweighing_{}.csv'.format(self.data_name), mode='a+', newline='',
                encoding='utf-8')
            self.data_writer_baseline_reweighing = csv.writer(self.data_file_baseline_reweighing)

            self.data_file_baseline_di_remover = open(self.new_path + 'Baseline_DIR_{}.csv'.format(self.data_name),
                                                      mode='a+', newline='', encoding='utf-8')
            self.data_writer_baseline_di_remover = csv.writer(self.data_file_baseline_di_remover)
        # todo: close csv here
        self.data_file_baseline.close()
        self.data_file_baseline_fairsmote.close()
        self.data_file_baseline_reweighing.close()
        self.data_file_baseline_di_remover.close()





if __name__ == '__main__':
    path = '../../../raw-data/'

    path_output = '../dataset-original/'
    ### Adult dataset
    # target_column = 'Probability'
    fair_error = 0.01
    alpha = 0.3
    correlation_threshold = 0.45  # 0.35
    loadData = LoadData(path, threshold=correlation_threshold)  # ,threshold=correlation_threshold

    #data_name, attrib = 'Student', [['Medu'],['Pstatus'], ['health'], ['activities'], ['absences'], ['sex']]  # less bia,

    #data_name, attrib = 'compas', [['race'], ['priors_count'], ['sex'], ['c_charge_degree'], ['age']] #MM LL ML LM

    data_name, attrib = 'bank', [['duration'], ['loan'], ['housing']] #MM ML LM

    #data_name, attrib = 'clevelan_heart', [['ca'], ['thal'], ['restecg'], ['age'], ['oldpeak']]  # MM,ll,ml,lm  ,  _35_threshold 'sex'


    # df_adult = loadData.load_adult_data('adult.data.csv')
    # df_adult = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    #df_adult = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
    # df_adult = loadData.load_student_data('Student.csv') #, drop_feature=['Pstatus', 'nursery']
    # df_adult = loadData.load_german_data2('german_credit_data.csv')
    # df_adult = loadData.load_german_data('GermanData.csv')
    # df_adult = loadData.load_compas_data('compas-scores-two-years.csv')
    df_adult = loadData.load_bank_data('bank.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_adult.columns.tolist()

    target_name = loadData.target_name
    target_index = loadData.target_index

    print(target_name, colums_list, df_adult)
    for colum in colums_list:
        print(set(df_adult[target_name].values.tolist()))
    sensitivity = Sensitive(df_adult, target_index, target_name, data_name,colums_list,path_output, alpha)
    sensitivity.fit(attrib)