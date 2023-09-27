import os
import time

import numpy as np
import pandas as pd
import csv
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import random
from src.common.metrics_utils import FairnessMetrics_partioning
from src.common.partial_ordering import Partial_ordering, VALID_TYPE_FREQUENT, VALID_TYPE_DEFINED
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical
from src.common.distance_measure import DistanceMeasure_specific
from src.common.models import Models, Models2
from src.common.load_data import LoadData
from src.common.sensitivity_utils import *
from src.common.utility_functions import *
from src.detection.swapping import Swapping


class Sensitive:
    def __init__(self, df_data, data, target_index, data_name, colums_list, log_path, threshold=1):
        self.df_data = df_data
        self.data = data
        self.target_index = target_index
        self.data_name = data_name
        self.colums_list = colums_list
        self.log_path = log_path
        self.threshold = threshold
        self.init_csv()

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

        self.log_path_local = self.new_path + 'Divergence_improved_2_local_{}.csv'.format(self.data_name)
        self.data_file_2 = open(self.log_path_local, mode='w', newline='',
                                encoding='utf-8')
        self.data_writer2 = csv.writer(self.data_file_2)
        self.data_writer2.writerow(
            ['ID', 'ID2', 'Feature', 'Feature2', 'Partial_ordering', 'swap_proportion', 'Type', 'P_a_b',
             'P_a_not_b', 'P_b_not_a', 'P_a_b_over_P_a_not_b', 'P_b_a_over_P_b_not_a',
             'Acc', 'Acc_after',  'hellinger_div', 'wasserstein_div',
              'total_variation_div', 'KL_div', 'JS_Div', 'Casuality', 'Importance', 'TPR_diff','TPR_diff_after',
             'FPR_diff', 'FPR_diff_after', 'SPD', 'SPD_after', 'DIR', 'DIR_after', 'AOD', 'AOD_after', 'TPR', 'TPR_after', 'FPR',
             'FPR_after', 'SP', 'SP_after', 'Pre', 'Pre_after', 'Re', 'Re_after', 'F1', 'F1_after', 'ACC', 'ACC_after'])

        self.log_path_shap = self.new_path + 'Shap-values_importance_{}-revision.csv'.format(data_name)
        self.data_file_3 = open(self.log_path_shap, mode='w', newline='',
                                encoding='utf-8')
        self.data_writer3 = csv.writer(self.data_file_3)
        self.data_writer3.writerow(['Fold', 'swap_proportion', 'Feature', 'Shap', 'Importance'])

    def _random_select_indices(self, y, proportion_):
        y_indices = [i for i in range(len(y))]
        N = int(round(len(y)*proportion_, 0))
        #print(y_indices, N)
        random.seed(10)
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

    def fit(self, ordering={}, swap_proportion=[0.1, 0.3, 0.5, 0.7], dmax=0.3):
        random.seed(10)
        X, Y = split_features_target(self.data, index=self.target_index)
        kf = model_selection.KFold(n_splits=10)
        data_partioning_dict = {}
        for column_id in range(X.shape[1]):
            data_partioning_dict[column_id] = data_partioning(self.data, column_id)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X=self.df_data,y=Y)):
            train, test = self.df_data.loc[train_idx], self.df_data.loc[test_idx]
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
            #explainer = shap.TreeExplainer(NNmodel)
            time_a = time.time()
            explainer = shap.LinearExplainer(NNmodel, self.x_train)
            #time_b = time.time()
            print("--- %s seconds ---" % (time.time() - time_a))
            p_pred = NNmodel.predict(self.x_test)

            dict_fairness_metrics = {}
            for column_id in range(self.x_test.shape[1]):
                column_data = get_features_by_index(self.x_test, column_id)
                (protected_1, non_protected_1) = self.__get_protected_attrib(column_id)
                fairnessMetrics_partioning = FairnessMetrics_partioning(column_data,data_partioning_dict[column_id],protected_1, non_protected_1,self.fav, self.non_fav)
                dict_fairness_metrics[column_id] = fairnessMetrics_partioning.fairness_metrics(self.y_test, p_pred)

            ## Call and run baseline here.
            ## TODO: LRTD
            acc_ = accuracy_score(p_pred, self.y_test)
            #baseline_lrtd = BaselineLRTD(train,test,colums_list, self.target_index, acc_, dict_fairness_metrics)
            #baseline_lrtd.fit_baseline(clf, p_pred, fold, self.data_writer_baseline)
            ## TODO: FairSMOTE
            #baseline_fairsmote = BaselineFairSMOTE(train, test, colums_list, self.target_index, acc_, dict_fairness_metrics)
            #baseline_fairsmote.fit_baseline(clf, p_pred, fold, self.data_writer_baseline_fairsmote)

            category_indices = []
            for column_id in range(self.x_train.shape[1]):
                if is_categorical(self.x_test, column_id):
                    category_indices.append(column_id)
            swapping = Swapping(category_indices)
            for swap_ratio in swap_proportion:
                print("********* ", swap_ratio)
                swap_perc = swap_ratio*100
                random_I = self._random_select_indices(self.y_test, swap_ratio)
                for column_id in range(self.x_test.shape[1]):

                    x_sampled_n = []
                    for i in range(len(self.x_test)):
                        if i in random_I:
                            x_sampled_n.append(self.x_test[i])

                    x_test_swapped = swapping.data_swapping(self.x_test, column_id,random_I, choice_dict=data_partioning_dict[column_id], dmax=dmax)

                    q_pred = NNmodel.predict(x_test_swapped)
                    #print(NNmodel.coef_)
                    #TODO: RUN SHAP Value Here..!
                    shap_values = explainer.shap_values(np.array(x_sampled_n))
                    vals = np.abs(shap_values).mean(0)
                    importance = NNmodel.coef_[0]
                    for i, v in enumerate(importance):
                        self.data_writer3.writerow([fold, swap_ratio, self.colums_list[i], vals[i], np.abs(v)])
                    ## TODO: Great! Done with Shap value for the current iteration
                    TPR_difference, FPR_difference, SPD, DIR, AOD, TPR, FPR, SP, Precision, Recal, F1, ACC = dict_fairness_metrics[column_id]
                    TPR_diff_after, FPR_diff_after, SPD_after, DIR_after, AOD_after, TPR_after, FPR_after, SP_after, Precision_after, Recal_after, F1_after, ACC_after = fairnessMetrics_partioning.fairness_metrics(self.y_test, q_pred)

                    print(column_id, Precision_after, Recal_after, F1_after, ACC_after)

                    kl_mean, js_mean = DistanceMeasure_specific.js_divergence(p_pred, q_pred)
                    hellinger_mean = DistanceMeasure_specific.hellinger_continous_1d(p_pred, q_pred)
                    wasserstein_mean = DistanceMeasure_specific.wasserstein_distance(p_pred, q_pred)
                    total_variation_mean = DistanceMeasure_specific.total_variation_distance(p_pred, q_pred)
                    acc_after = accuracy_score(q_pred, self.y_test)

                    f_importance = round(np.sum(self.y_test != q_pred)*100/len(self.y_test), 3)
                    proportion_ = round(np.sum(p_pred != q_pred)*100/len(self.y_test), 3)

                    self.data_writer2.writerow(
                        ['F_{}'.format(column_id), 'F_{}'.format(column_id), self.colums_list[column_id], self.colums_list[column_id],'N/A', swap_perc, 'CDI',
                         0, 0, 0, 0, 0,
                         acc_, acc_after, hellinger_mean,
                         wasserstein_mean, total_variation_mean, kl_mean, js_mean, proportion_, f_importance,
                         TPR_difference, TPR_diff_after, FPR_difference, FPR_diff_after, SPD, SPD_after, DIR, DIR_after, AOD, AOD_after,
                         TPR, TPR_after, FPR, FPR_after, SP, SP_after, Precision, Precision_after, Recal, Recal_after, F1, F1_after, ACC,
                         ACC_after])
                    for column_id2 in range(self.x_test.shape[1]):
                        if column_id != column_id2:
                            valid_order, type_ = Partial_ordering._ordering(ordering=ordering,f1_feature=self.colums_list[column_id], f2_mediator=self.colums_list[column_id2])
                            if self.colums_list[column_id] == 'sex':
                                print('   ---- ', self.colums_list[column_id], self.colums_list[column_id2], valid_order, type_)
                            if valid_order:
                                flag_ordering = False
                                if type_ == VALID_TYPE_FREQUENT:
                                    ordering_input_type = VALID_TYPE_FREQUENT
                                    (P_a_b, P_a_not_b, P_b_not_a, P_a_b_over_P_a_not_b, P_b_a_over_P_b_not_a) = self.prob_ordering_freq(column_id, column_id2, data_partioning_dict)
                                    if P_a_b_over_P_a_not_b > P_b_a_over_P_b_not_a:
                                        flag_ordering = True
                                else:
                                    flag_ordering = True
                                    ordering_input_type = VALID_TYPE_DEFINED
                                    (P_a_b, P_a_not_b, P_b_not_a, P_a_b_over_P_a_not_b, P_b_a_over_P_b_not_a) = (0,0,0,0,0)

                                ## TODO: Natural Indirect impact
                                if flag_ordering:
                                    x_test_swapped2 = swapping.double_swapping(x_test_swapped,self.x_test, column_id, column_id2,random_I, choice_dict=data_partioning_dict[column_id2],dmax=dmax)
                                    q_pred_both = NNmodel.predict(x_test_swapped2)

                                    TPR_diff_after, FPR_diff_after, SPD_after, DIR_after, AOD_after, TPR_after, FPR_after, SP_after, Precision_after, Recal_after, F1_after, ACC_after = fairnessMetrics_partioning.fairness_metrics(
                                        q_pred, q_pred_both)

                                    kl_mean, js_mean = DistanceMeasure_specific.js_divergence(q_pred, q_pred_both)
                                    hellinger_mean = DistanceMeasure_specific.hellinger_continous_1d(q_pred, q_pred_both)
                                    wasserstein_mean = DistanceMeasure_specific.wasserstein_distance(q_pred, q_pred_both)
                                    total_variation_mean = DistanceMeasure_specific.total_variation_distance(q_pred, q_pred_both)
                                    acc_after = accuracy_score(q_pred, q_pred_both)

                                    f_importance = round(np.sum(q_pred_both != q_pred) * 100 / len(self.y_test), 3)
                                    proportion_ = round(np.sum(p_pred != q_pred_both) * 100 / len(self.y_test), 3)

                                    print('NII: ', kl_mean, hellinger_mean, wasserstein_mean, total_variation_mean, acc_after, f_importance, TPR_diff_after, FPR_diff_after, SPD_after, DIR_after, AOD_after, TPR_after, FPR_after, SP_after, Precision_after, Recal_after, F1_after, ACC_after)
                                    self.data_writer2.writerow(
                                        ['F_{}'.format(column_id), 'F_{}'.format(column_id2), self.colums_list[column_id],
                                         self.colums_list[column_id2], ordering_input_type, swap_perc, 'NII',
                                         P_a_b, P_a_not_b, P_b_not_a, P_a_b_over_P_a_not_b, P_b_a_over_P_b_not_a,
                                         acc_, acc_after, hellinger_mean,
                                         wasserstein_mean, total_variation_mean, kl_mean, js_mean, proportion_,
                                         f_importance, TPR_difference, TPR_diff_after, FPR_difference, FPR_diff_after, SPD, SPD_after, DIR,
                                         DIR_after, AOD, AOD_after,
                                         TPR, TPR_after, FPR, FPR_after, SP, SP_after, Precision, Precision_after, Recal,
                                         Recal_after, F1, F1_after, ACC,
                                         ACC_after])
                                    ## TODO: Natural direct impact
                                    x_test_swapped_direct = swapping.data_swapping(self.x_test, column_id2, random_I,
                                                                            choice_dict=data_partioning_dict[column_id2],
                                                                            dmax=dmax)

                                    q_pred = NNmodel.predict(x_test_swapped_direct)

                                    x_test_swapped_direct2 = swapping.double_swapping(x_test_swapped_direct, self.x_test, column_id,
                                                                               column_id2, random_I,
                                                                               choice_dict=data_partioning_dict[column_id],
                                                                               dmax=dmax)
                                    q_pred_both = NNmodel.predict(x_test_swapped_direct2)

                                    TPR_diff_after, FPR_diff_after, SPD_after, DIR_after, AOD_after, TPR_after, FPR_after, SP_after, Precision_after, Recal_after, F1_after, ACC_after = fairnessMetrics_partioning.fairness_metrics(
                                        q_pred, q_pred_both)

                                    kl_mean, js_mean = DistanceMeasure_specific.js_divergence(q_pred, q_pred_both)
                                    hellinger_mean = DistanceMeasure_specific.hellinger_continous_1d(q_pred, q_pred_both)
                                    wasserstein_mean = DistanceMeasure_specific.wasserstein_distance(q_pred, q_pred_both)
                                    total_variation_mean = DistanceMeasure_specific.total_variation_distance(q_pred, q_pred_both)
                                    acc_after = accuracy_score(q_pred, q_pred_both)

                                    f_importance = round(np.sum(q_pred_both != q_pred) * 100 / len(self.y_test), 3)
                                    proportion_ = round(np.sum(p_pred != q_pred_both) * 100 / len(self.y_test), 3)

                                    print('NDI: ', kl_mean, hellinger_mean, wasserstein_mean, total_variation_mean,acc_after, f_importance, TPR_diff_after, FPR_diff_after, SPD_after,
                                          DIR_after, AOD_after, TPR_after, FPR_after, SP_after, Precision_after, Recal_after,
                                          F1_after, ACC_after)

                                    self.data_writer2.writerow(
                                        ['F_{}'.format(column_id), 'F_{}'.format(column_id2), self.colums_list[column_id],
                                         self.colums_list[column_id2], ordering_input_type, swap_perc, 'NDI',
                                         P_a_b, P_a_not_b, P_b_not_a, P_a_b_over_P_a_not_b, P_b_a_over_P_b_not_a,
                                         acc_, acc_after, hellinger_mean,
                                         wasserstein_mean, total_variation_mean, kl_mean, js_mean, proportion_,
                                         f_importance, TPR_difference, TPR_diff_after, FPR_difference, FPR_diff_after, SPD, SPD_after, DIR,
                                         DIR_after, AOD, AOD_after,
                                         TPR, TPR_after, FPR, FPR_after, SP, SP_after, Precision, Precision_after, Recal,
                                         Recal_after, F1, F1_after, ACC,
                                         ACC_after])

                                    # todo: close csv here
                                    self.data_file_2.close()
                                    self.data_file_3.close()

                                    self.data_file_2 = open(self.log_path_local, mode='a+', newline='',
                                                            encoding='utf-8')
                                    self.data_writer2 = csv.writer(self.data_file_2)

                                    ## todo Sharp value
                                    self.data_file_3 = open(self.log_path_shap, mode='a+', newline='',
                                                            encoding='utf-8')
                                    self.data_writer3 = csv.writer(self.data_file_3)

        #todo: close csv here
        self.data_file_2.close()
        self.data_file_3.close()





if __name__ == '__main__':
    path = '../../raw-data/'


    path_output = '../../dataset-default/'

    ## TODO: Please specify the hyperparams here
    fair_error = 0.01 # Optional
    alpha = 0.3 # distortion
    correlation_threshold = None #0.45  # remove correlated features
    loadData = LoadData(path, threshold=correlation_threshold)
    #TODO; Uncomment the line before, for the dataset you want to analyse
    ## Its always optional to provide the partial ordering, either part full or part. Leave the dictionary empy otherwise
    #data_name, ordering = 'Student', {0:['sex','age']}
    #data_name, ordering = 'clevelan_heart', {0:['sex']} # , 'age'
    # data_name, ordering = 'bank',  {}#
    data_name, ordering = 'compas', {0:['race', 'sex', 'age']}

    # df_data = loadData.load_adult_data('adult.data.csv')
    # df_data = loadData.load_credit_defaulter('default_of_credit_card_clients.csv')
    #df_data = loadData.load_clevelan_heart_data('processed.cleveland.data.csv')
    # df_data = loadData.load_student_data('Student.csv') #, drop_feature=['Pstatus', 'nursery']
    # df_data = loadData.load_german_data2('german_credit_data.csv')
    # df_data = loadData.load_german_data('GermanData.csv')
    df_data = loadData.load_compas_data('compas-scores-two-years.csv')
    # df_data = loadData.load_bank_data('bank.csv')
    sensitive_list = loadData.sensitive_list
    sensitive_indices = loadData.sensitive_indices
    colums_list = df_data.columns.tolist()

    target_name = loadData.target_name
    target_index = loadData.target_index

    print(target_name, colums_list, df_data)
    for colum in colums_list:
        print(set(df_data[target_name].values.tolist()))
    sensitivity = Sensitive(df_data, df_data.to_numpy(),target_index,data_name,colums_list,path_output, alpha)
    sensitivity.fit(ordering=ordering, dmax=alpha)