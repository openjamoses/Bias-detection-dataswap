import operator

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import random

def split_features_target(data, index=20):
    #print(data[0:, 0:index], data[0:, index:])
    x= np.concatenate((data[0:, 0:index], data[0:, index+1:]), axis=1)
    print(x.shape)
    return x, data[0:, index]
    #return data[0:, 0:index]+data[0:, index:], data[0:, index]
def data_split(data, sample_size=0.25):
    return train_test_split(data, train_size=None, test_size=sample_size, random_state=42)
def scores_metrics(gda_pred, y_test):
    accuracy = round(metrics.accuracy_score(y_test, gda_pred),3)          # accuracy: (tp + tn) / (p + n)
    precision = round(metrics.precision_score(y_test, gda_pred),3)          # precision tp / (tp + fp)
    recall = round(metrics.recall_score(y_test, gda_pred) ,3)               # recall: tp / (tp + fn)
    f1 = round(metrics.f1_score(y_test, gda_pred) ,3)                       # f1: 2 tp / (2 tp + fp + fn)

    print('precision: ', precision, 'recall: ', recall, 'f1: ', f1,  'accuracy: ', accuracy)
    return precision, recall, f1, accuracy

def sort_dict(dict_, reverse=True):
    return dict(sorted(dict_.items(), key=operator.itemgetter(1), reverse=reverse))
def get_features_by_index(X, column_id):
    feature_data = np.array(X)[0:, column_id]
    return [val for val in feature_data]
def data_partioning(data, feature_index):
    #data_range = self.data[0:, feature_index]
    data_range = data[0:, feature_index]
    folded_data = {}
    unique_ = np.unique(data_range)
    if len(unique_) == 2:
        folded_data[0.0] = [np.unique(data_range)[0] if np.unique(data_range)[0] < np.unique(data_range)[1] else np.unique(data_range)[1]]
        folded_data[1.0] = [np.unique(data_range)[0] if np.unique(data_range)[0] > np.unique(data_range)[1] else \
        np.unique(data_range)[1]]
    elif len(unique_) > 2 and len(unique_) <= 6:
        medium = np.median(unique_)
        folded_data[0.0] = [v for v in unique_ if v <= medium]
        folded_data[1.0] = [v for v in unique_ if v > medium]
    else:
        #print(data_range)
        #percentile_50_ = (min(data_range) + max(data_range))/2 #np.percentile(list(set(data_range)), 50)
        percentile_50_ = np.percentile(unique_, 50)
        # percentile_50_ = np.mean(data_range)
        percentile_75 = np.percentile(data_range, 75)
        percentile_50 = max([i for i in unique_ if i <= percentile_50_])
        percentile_100 = np.percentile(data_range, 100)
        #print('percentile_50: ', percentile_50, percentile_100, np.unique(data_range), data_range)
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
            if data_range[i] <= percentile_50:
                fold_id = 0.0
                #data[0:, feature_index] = np.where(data[0:, feature_index] == data_range[i], fold_id, data[0:, feature_index])
            elif data_range[i] > percentile_50:  # and data_range[i] <= percentile_50:
                fold_id = 1.0
            #data[0:, feature_index] = np.where(data[0:, feature_index] == data_range[i], fold_id,
            #                                        data[0:, feature_index])
            if fold_id in folded_data.keys():
                folded_data[fold_id].add(fold_id)
            else:
                folded_data[fold_id] = set([data_range[i]])
        for key, val in folded_data.items():
            if len(val) < 3:
                if key == 0.0:
                    val2 = []
                    val2.append(random.uniform(np.min(list(val)), np.max(list(val))))
                    val2.append(random.uniform(np.min(list(val)), np.max(list(val))))
                    val2.extend(list(val))
                if key == 1.0:
                    val2 = []
                    val2.append(random.uniform(percentile_50, np.max(list(val))))
                    val2.append(random.uniform(percentile_50, np.max(list(val))))
                    val2.extend(list(val))
                folded_data[key] = list(val)
            else:
                folded_data[key] = list(val)
    return folded_data
def subclass_probablity(x_test, y_test, y_pred, sensitive_index):
    data = {}
    for i in range(len(x_test)):
        if x_test[i][sensitive_index] in data.keys():
            if y_pred[i] in data[x_test[i][sensitive_index]].keys():
                if y_test[i] in data[x_test[i][sensitive_index]][y_pred[i]].keys():
                    data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] += 1
                else:
                    data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = 1
            else:
                data[x_test[i][sensitive_index]][y_pred[i]] = {}
                data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = 1
        else:
            data[x_test[i][sensitive_index]] = {}
            data[x_test[i][sensitive_index]][y_pred[i]] = {}
            data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = 1

        #data[x_test[i][sensitive_index]][y_pred[i]][y_test[i]] = data.get([x_test[i][sensitive_index]][y_pred[i]][y_test[i]], 0) + 1
    return data

def transform_df_minmax(df_data):
    scaler = MinMaxScaler()
    df_data = pd.DataFrame(scaler.fit_transform(df_data), columns=df_data.columns)
    #dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, shuffle=True)
    return df_data

def fairness_metrics(x_test, y_test, y_pred, sensitive_index, protected=0, unprotected=1):
    data = subclass_probablity(x_test,y_test,y_pred,sensitive_index)
    AOD = calculate_average_odds_difference(data, protected=protected,unprotected=unprotected)
    DI = calculate_Disparate_Impact(data, protected=protected, unprotected=unprotected)
    DIR = calculate_Disparate_Impact_ratio(data, protected=protected, unprotected=unprotected)
    SPD = calculate_SPD(data, protected=protected, unprotected=unprotected)
    EOD = calculate_equal_opportunity_difference(data, protected=protected, unprotected=unprotected)
    TPR = calculate_TPR_difference(data, protected=protected, unprotected=unprotected)
    FPR = calculate_FPR_difference(data, protected=protected, unprotected=unprotected)
    print('AOD: ', AOD, 'DI: ', DI, 'DIR: ', DIR, 'SPD: ', SPD, 'EOD: ', EOD, 'TPR: ', TPR, 'FPR: ', FPR)
    return AOD, DI, DIR, SPD, EOD, TPR, FPR

def calculate_average_odds_difference(data , protected=0, unprotected=1):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)
    # FPR_male = FP_male/(FP_male+TN_male)
    # FPR_female = FP_female/(FP_female+TN_female)
    # average_odds_difference = abs(abs(TPR_male - TPR_female) + abs(FPR_male - FPR_female))/2
    #FPR_diff = calculate_FPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    #TPR_diff = calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    FPR_diff = calculate_FPR_difference(data=data,protected=protected, unprotected=unprotected)
    TPR_diff = calculate_TPR_difference(data=data,protected=protected, unprotected=unprotected)

    average_odds_difference = (FPR_diff + TPR_diff)/2
    #print("average_odds_difference",average_odds_difference)
    return round(average_odds_difference,3)
def extract_stats(data , protected=0, unprotected=1):
    TP_1 = 0
    FP_1 = 0
    TN_1 = 0
    FN_1 = 0

    FN_0 = 0
    FP_0 = 0
    TP_0 = 0
    TN_0 = 0
    if unprotected in data.keys():
        if 0 in data[unprotected].keys():
            if 0 in data[unprotected][0].keys():
                TN_1 = data[unprotected][0][0]
            if 1 in data[unprotected][0].keys():
                FN_1 = data[unprotected][0][1]
        if 1 in data[unprotected].keys():
            if 1 in data[unprotected][1].keys():
                TP_1 = data[unprotected][1][1]
            if 0 in data[unprotected][1].keys():
                FP_1 = data[unprotected][1][0]
    if protected in data.keys():
        if 0 in data[protected].keys():
            if 0 in data[protected][0].keys():
                TN_0 = data[protected][0][0]
            if 1 in data[protected][0].keys():
                FN_0 = data[protected][0][1]
        if 1 in data[protected].keys():
            if 1 in data[protected][1].keys():
                TP_0 = data[protected][1][1]
            if 0 in data[protected][1].keys():
                FP_0 = data[protected][1][0]

    return TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0

def calculate_Disparate_Impact(data , protected=0, unprotected=1):
    TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0 = extract_stats(data,protected,unprotected)
    #P1 = (data.get(unprotected,0).get(1,0)+data[unprotected][0][1])/(data[unprotected][1][1]+data[unprotected][0][0]+
    #                                                        data[unprotected][1][0]+data[unprotected][0][1])
    #P0 = (data[protected][1][1] + data[protected][0][1]) / (data[protected][1][1] + data[protected][0][0] +
    #                                                       data[protected][1][0] + data[protected][0][1])


    P1 = (TP_1 + FP_1)/(TP_1 + TN_1 + FN_1 + FP_1)
    P0 = (TP_0 + FP_0)/(TP_0 + TN_0 + FN_0 +  FP_0)
    DI = (P0/P1)
    return round((1 - abs(DI)),3)

def calculate_Disparate_Impact_ratio(data , protected=0, unprotected=1):

    P1 = data[unprotected][1][1]/np.sum(list(data[unprotected][1].values()))
    P0 = data[protected][1][1] / np.sum(list(data[protected][1].values()))
    #P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
    #P_female =  (TP_female + FP_female)/(TP_female + TN_female + FN_female +  FP_female)
    DI = (P0/P1)
    return round(DI,3)

def calculate_SPD(data , protected=0, unprotected=1):
    #P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
    #P_female = (TP_female + FP_female) /(TP_female + TN_female + FN_female +  FP_female)
    TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0 = extract_stats(data, protected, unprotected)
    #P1 = (data[unprotected][1][1] + data[unprotected][0][1]) / (data[unprotected][1][1] + data[unprotected][0][0] +
    #                                                            data[unprotected][1][0] + data[unprotected][0][1])
    #P0 = (data[protected][1][1] + data[protected][0][1]) / (data[protected][1][1] + data[protected][0][0] +
    #                                                        data[protected][1][0] + data[protected][0][1])
    P1 = (TP_1 + FP_1)/(TP_1 + TN_1 + FN_1 + FP_1)
    P0 = (TP_0 + FP_0) /(TP_0 + TN_0 + FN_0 +  FP_0)
    SPD = (P0 - P1)
    return round(abs(SPD),3)
def calculate_TPR(TP, FP, TN,FN):
    #TPR = TP/(TP+FN)
    TPR = 0
    if (TP+FN) > 0:
        TPR = TP/(TP+FN)
    return round(TPR,3)
def calculate_FPR(TP, FP, TN,FN):
    #FPR = FP/(FP+TN)
    FPR = 0
    if (FP+TN) > 0:
        FPR = FP/(FP+TN)
    return round(FPR,3)

def calculate_SP(TP, FP, TN,FN):
    SP = 0
    if (TP + TN + FN + FP) > 0:
        SP = (TP + FP) / (TP + TN + FN + FP)
    return round(SP,3)
def calculate_proportion(TP, FP, TN,FN):
    Probability = 0
    if (TP + TN + FN + FP) > 0:
        Probability = (TP + FP) / (TP + TN + FN + FP)
    return round(Probability,3)
# Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
# Precision Score = TP / (FP + TP)
# Recall Score = TP / (FN + TP)
# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)

def calculate_Precision(TP, FP, TN,FN):
    Precision = 0
    if (TP + FP) > 0:
        Precision = TP / (TP + FP)
    return round(Precision,3)
def calculate_Recall(TP, FP, TN,FN):
    Recall = 0
    if (TP + FN) > 0:
        Recall = TP / (TP + FN)
    return round(Recall,3)

def calculate_Accuracy(TP, FP, TN,FN):
    accuracy = 0
    if (TP + FN + TN + FP) > 0:
        accuracy = (TP +TN) / (TP + FN + TN + FP)
    return round(accuracy,3)

def calculate_F1(TP, FP, TN,FN):
    precision = calculate_Precision(TP, FP, TN,FN)
    recall = calculate_Recall(TP, FP, TN, FN)
    F1 = 0
    if (precision+recall) > 0:
        F1 = (2*precision*recall) / (precision+recall)
    return round(F1,3)

def calculate_Accuracy(TP, FP, TN,FN):
    accuracy = 0
    if (TP + FP) > 0:
        accuracy = (TP +TN) / (TP + FN + TN + FP)
    return round(accuracy,3)
def calculate_equal_opportunity_difference(data , protected=0, unprotected=1):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)
    # equal_opportunity_difference = abs(TPR_male - TPR_female)
    #print("equal_opportunity_difference:",equal_opportunity_difference)
    return calculate_TPR_difference(data,protected=protected, unprotected=unprotected)
    #return calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)

def calculate_TPR_difference(data , protected=0, unprotected=1):
    #TPR_male = TP_male/(TP_male+FN_male)
    #TPR_female = TP_female/(TP_female+FN_female)
    TPR_male = data[unprotected][1][1] / (data[unprotected][1][1] + data[unprotected][1][0])
    TPR_female = data[protected][1][1] / (data[protected][1][1] + data[protected][1][0])
    # print("TPR_male:",TPR_male,"TPR_female:",TPR_female)
    diff = (TPR_male - TPR_female)
    return round(diff,3)

def calculate_FPR_difference(data , protected=0, unprotected=1):
    #FPR_male = FP_male/(FP_male+TN_male)
    #FPR_female = FP_female/(FP_female+TN_female)
    TP_1, FP_1, TN_1, FN_1, TP_0, FP_0, TN_0, FN_0 = extract_stats(data, protected, unprotected)

    FPR_1 = FP_1 / (FP_1 + TN_1)
    FPR_0 = FP_0 / (FP_0 + TN_0)
    # print("FPR_male:",FPR_male,"FPR_female:",FPR_female)
    diff = (FPR_0 - FPR_1)
    return round(diff,3)

def compute_fairness_metrics(column_data, choice_dict, y_original, y_predicted, protected_key=0.0, non_protected_key=1.0, fav=1.0, non_fav=0.0):
    TP_1 = 0
    FP_1 = 0
    TN_1 = 0
    FN_1 = 0

    FN_0 = 0
    FP_0 = 0
    TP_0 = 0
    TN_0 = 0
    for i in range(len(column_data)):
        if column_data[i] in choice_dict[protected_key]:
            if y_predicted[i] == fav and y_predicted[i] == y_original[i]:
                TP_0 += 1
            elif y_predicted[i] == fav and y_predicted[i] != y_original[i]:
                FP_0 += 1
            elif y_predicted[i] == non_fav and y_predicted[i] == y_original[i]:
                TN_0 += 1
            elif y_predicted[i] == non_fav and y_predicted[i] != y_original[i]:
                FN_0 += 1
        elif column_data[i] in choice_dict[non_protected_key]:
            if y_predicted[i] == fav and y_predicted[i] == y_original[i]:
                TP_1 += 1
            elif y_predicted[i] == fav and y_predicted[i] != y_original[i]:
                FP_1 += 1
            elif y_predicted[i] == non_fav and y_predicted[i] == y_original[i]:
                TN_1 += 1
            elif y_predicted[i] == non_fav and y_predicted[i] != y_original[i]:
                FN_1 += 1
def fairness_metrics(TN, FP, FN, TP):
    TPR = calculate_TPR(TP, FP, TN, FN)
    FPR = calculate_FPR(TP, FP, TN, FN)
    SP = calculate_SP(TP, FP, TN, FN)

    Precision = calculate_Precision(TP, FP, TN, FN)
    Recal = calculate_Recall(TP, FP, TN, FN)
    F1 = calculate_F1(TP, FP, TN, FN)
    ACC = calculate_Accuracy(TP, FP, TN, FN)
    return TPR, FPR, SP, Precision, Recal, F1, ACC
