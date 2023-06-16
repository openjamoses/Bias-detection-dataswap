import csv

import pandas as pd
import os
import numpy as np
# from scipy.stats import ranksums, pearsonr
from scipy.stats import ranksums, pearsonr, chi2_contingency
from scipy import stats


# Pass two "LISTS" to the function to calculate the difference in Cliff's delta.
def cliffDelta(x, y, decimals=2):
    lenx = len(x)
    leny = len(y)

    ## generate a matrix full of zeros
    matrix = np.zeros((lenx, leny))

    ## compare the two lists and put either 1 or -1 to the matrix (if they are equal, there is already a zero in the matrix)
    for i in range(lenx):
        for j in range(leny):
            if x[i] > y[j]:
                matrix[i, j] = 1
            elif x[i] < y[j]:
                matrix[i, j] = -1

    ## get the avarage of the dominance matrix
    delta = matrix.mean()
    return round(delta, decimals), matrix


def ranking(p_value, delta):
    W, T, L = 0, 0, 0
    Rank = ''
    if p_value < 0.05 and delta > 0.147:
        W += 1
        Rank = 'W'
    elif p_value < 0.05 and delta > -0.147:
        L += 1
        Rank = 'L'
    else:
        T += 1
        Rank = 'T'
    return Rank


def create_dict(feature, droped_attrib, metric, val_before, val_after, data_before, data_after):
    if not feature in data_before.keys():
        data_before[feature] = {}
        data_after[feature] = {}
    if metric in data_before[feature].keys():
        if droped_attrib in data_before[feature][metric].keys():
            data_before[feature][metric][droped_attrib].append(val_before)
            data_after[feature][metric][droped_attrib].append(val_after)
        else:
            data_before[feature][metric][droped_attrib] = [val_before]
            data_after[feature][metric][droped_attrib] = [val_after]
    else:
        data_before[feature][metric] = {}
        data_before[feature][metric][droped_attrib] = [val_before]
        data_after[feature][metric] = {}
        data_after[feature][metric][droped_attrib] = [val_after]


def generate_ranking(data_before, data_after, PERFORMANCE):
    Rank_dict = {}
    new_size = 100

    gloabL_data_before = {}
    gloabL_data_after = {}
    for f, v in data_before.items():
        for key, val in data_before[f].items():
            for key2, val2 in val.items():
                # accuracy
                val_before = val2
                val_after = data_after[f][key][key2]

                if key in gloabL_data_before.keys():
                    if key2 in gloabL_data_before[key].keys():
                        gloabL_data_before[key][key2].append(val_before)
                        gloabL_data_after[key][key2].append(val_after)
                    else:
                        gloabL_data_before[key][key2] = [val_before]
                        gloabL_data_after[key][key2] = [val_after]
                else:
                    gloabL_data_before[key] = {}
                    gloabL_data_before[key][key2] = [val_before]

                    gloabL_data_after[key] = {}
                    gloabL_data_after[key][key2] = [val_after]

                # Creating a contingency table
                # cont_table = pd.crosstab(index=val_before,
                #                          columns=val_after)

                # # Chi-square value
                # X2 = chi2_contingency(cont_table)
                # chi_stat = X2[0]
                #
                # # Size of the sample
                # N = len(val_after)
                #
                # minimum_dimension = (min(cont_table.shape) - 1)
                #
                # # Calculate Cramer's V
                # result = np.sqrt((chi_stat / N) / minimum_dimension)

                # val_before = np.resize(val_before, new_size)
                # val_after = np.resize(val_after, new_size)

                if key in PERFORMANCE:
                    # val_after =  np.resize(data_after[f][key][key2], new_size)
                    # val_before = np.resize(val2, new_size)

                    _stats, _pvalue = ranksums(val_after, val_before)
                    _delta, _ = cliffDelta(val_after, val_before)
                    rank_val = ranking(_pvalue, _delta)

                    # print(key, key2, _stats, _pvalue, _delta, rank_val)
                else:
                    _stats, _pvalue = ranksums(val_before, val_after)
                    _delta, _ = cliffDelta(val_before, val_after)
                    rank_val = ranking(_pvalue, _delta)
                    # print(key, key2, _stats, _pvalue, _delta, rank_val)
                if key2 in Rank_dict.keys():
                    if key in Rank_dict[key2].keys():
                        if rank_val in Rank_dict[key2][key].keys():
                            Rank_dict[key2][key][rank_val] += 1
                        else:
                            Rank_dict[key2][key][rank_val] = 1
                    else:
                        Rank_dict[key2][key] = {}
                        Rank_dict[key2][key][rank_val] = 1
                else:
                    Rank_dict[key2] = {}
                    Rank_dict[key2][key] = {}
                    Rank_dict[key2][key][rank_val] = 1
    for key, val in gloabL_data_after.items():
        for key2, val2 in val.items():
            val_before = gloabL_data_before[key][key2]
            val_after = gloabL_data_after[key][key2]
            # vals, count = stats.contingency.crosstab(val_before, val_after)
            # assoc_2 = stats.contingency.association(count, method="pearson")
            # print(key, key2, assoc_2)

    return Rank_dict


# todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
path_original = '../dataset-original/'
# path_filtered = '../dataset-filtered/'
# if __name__ == '__main__':
# df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
# data_name = 'adult'
# data_name = 'clevelan_heart'
#data_name = 'clevelan_heart-clean'
# data_name = 'Student-new'
# data_name = 'Student-new'
# data_name = 'compas-new2'
data_name = 'bank-clean'
#data_name = 'compas-clean'
# data_name = 'compas'
#data_name = 'Student-clean'
alpha = 0.3
naming = data_name  # '{}-{}_'.format(data_name, alpha)  # _35_threshold
path_output_ = '/Volumes/Cisco/Summer2023/Fairness/Revision/experiments2/{}/'.format(data_name)

BASELINES = ['LRTD', 'fairSMOTE', 'Reweighing', 'DIR']

path_output_ = '/Volumes/Cisco/Summer2023/Fairness/Revision/experiments2/{}/'.format(data_name)
# path_output2 = '/Volumes/Cisco/Fall2022/Fairness/Analysis/Ranking/Shap/'
if not os.path.exists(path_output_):
    os.makedirs(path_output_)
if not os.path.exists(path_output_ + '/Ranking'):
    os.makedirs(path_output_ + '/Ranking')
path_output = path_output_ + '/Ranking/'

ACC, PRE, RE, F1, FPR, AOD, SPD, DIR, FPR_diff = 'ACC', 'PRE', 'RE', 'F1', 'FPR', 'AOD', 'SPD', 'DIR', 'FPR_diff'

data_file = open(path_output + 'statistical-tests-{}.csv'.format(data_name),
                 mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(
    ['Baseline', 'Measures', 'Accuracy', 'Precision', 'Recall', 'F1', 'False-alarm', 'AOD', 'SPD', 'DIR', 'FPR_diff',
     'SUM_W', 'SUM_T', 'SUM_L'])

for baseline_name in BASELINES:

    print(' baseline: ', baseline_name)
    df_data = pd.read_csv(
        path_original + naming + '/' + str(alpha) + '/' + 'Baseline_{}_{}.csv'.format(baseline_name, naming))
    #df_lrtd = pd.read_csv(path_original + naming + '/' + str(alpha) + '/' + 'Baseline_LRTD_{}.csv'.format(naming))

    Feature_smote = df_data.Feature.values.tolist()
    Droped_Attrib = df_data.Droped_Attrib.values.tolist()
    acc_smote = df_data.acc_.values.tolist()
    acc_after_smote = df_data.acc_after.values.tolist()
    FPR_diff_smote = df_data.FPR_diff.values.tolist()
    FPR_diff_after_smote = df_data.FPR_diff_after.values.tolist()
    SPD_smote = df_data.SPD.values.tolist()
    SPD_after_smote = df_data.SPD_after.values.tolist()
    DIR_smote = df_data.DIR.values.tolist()
    DIR_after_smote = df_data.DIR_after.values.tolist()
    AOD_smote = df_data.AOD.values.tolist()
    AOD_after_smote = df_data.AOD_after.values.tolist()
    FPR_smote = df_data.FPR.values.tolist()
    FPR_after_smote = df_data.FPR_after.values.tolist()
    SP_smote = df_data.SP.values.tolist()
    SP_after_smote = df_data.SP_after.values.tolist()
    Pre_smote = df_data.Pre.values.tolist()
    Pre_after_smote = df_data.Pre_after.values.tolist()
    Re_smote = df_data.Re.values.tolist()
    Re_after_smote = df_data.Re_after.values.tolist()

    F1_smote = df_data.F1.values.tolist()
    F1_after_smote = df_data.F1_after.values.tolist()
    data_before = {}
    data_after = {}
    for i in range(len(Feature_smote)):
        # todo: accuracy
        feature = Feature_smote[i]
        droped_attrib = Droped_Attrib[i]
        # todo: Accuracy
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=ACC, val_before=acc_smote[i],
                    val_after=acc_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: Precision
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=PRE, val_before=Pre_smote[i],
                    val_after=Pre_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: Recall
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=RE, val_before=Re_smote[i],
                    val_after=Re_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: F1-score
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=F1, val_before=F1_smote[i],
                    val_after=F1_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: FPR
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=FPR, val_before=FPR_smote[i],
                    val_after=FPR_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: AOD
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=AOD, val_before=AOD_smote[i],
                    val_after=AOD_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: SPD
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=SPD, val_before=SPD_smote[i],
                    val_after=SPD_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: DIR
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=DIR, val_before=DIR_smote[i],
                    val_after=DIR_after_smote[i], data_before=data_before, data_after=data_after)
        # todo: FPR_difference
        create_dict(feature=feature, droped_attrib=droped_attrib, metric=FPR_diff, val_before=FPR_diff_smote[i],
                    val_after=FPR_diff_after_smote[i], data_before=data_before, data_after=data_after)

    PERFORMANCE = [ACC, PRE, RE, F1]
    Rank_dict = generate_ranking(data_before=data_before, data_after=data_after, PERFORMANCE=PERFORMANCE)

    for key, val in Rank_dict.items():
        V_ACC, V_PRE, V_RE, V_F1, V_FPR, V_AOD, V_SPD, V_DIR, V_FPR_diff = Rank_dict[key][ACC], Rank_dict[key][PRE], \
        Rank_dict[key][RE], Rank_dict[key][F1], Rank_dict[key][FPR], Rank_dict[key][AOD], Rank_dict[key][SPD], \
        Rank_dict[key][DIR], Rank_dict[key][FPR_diff]

        SUM_W = V_ACC.get('W', 0) + V_PRE.get('W', 0) + V_RE.get('W', 0) + V_F1.get('W', 0) + V_FPR.get('W', 0) + V_AOD.get(
            'W', 0) + V_SPD.get('W', 0) + V_DIR.get('W', 0) + V_FPR_diff.get('W', 0)
        SUM_T = V_ACC.get('T', 0) + V_PRE.get('T', 0) + V_RE.get('T', 0) + V_F1.get('T', 0) + V_FPR.get('T', 0) + V_AOD.get(
            'T', 0) + V_SPD.get('T', 0) + V_DIR.get('T', 0) + V_FPR_diff.get('T', 0)
        SUM_L = V_ACC.get('L', 0) + V_PRE.get('L', 0) + V_RE.get('L', 0) + V_F1.get('L', 0) + V_FPR.get('L', 0) + V_AOD.get(
            'L', 0) + V_SPD.get('L', 0) + V_DIR.get('L', 0) + V_FPR_diff.get('L', 0)

        data_writer.writerow(
            [baseline_name, key, '{}|{}|{}'.format(V_ACC.get('W', 0), V_ACC.get('T', 0), V_ACC.get('L', 0)),
             '{}|{}|{}'.format(V_PRE.get('W', 0), V_PRE.get('T', 0), V_PRE.get('L', 0)),
             '{}|{}|{}'.format(V_RE.get('W', 0), V_RE.get('T', 0), V_RE.get('L', 0)),
             '{}|{}|{}'.format(V_F1.get('W', 0), V_F1.get('T', 0), V_F1.get('L', 0)),
             '{}|{}|{}'.format(V_FPR.get('W', 0), V_FPR.get('T', 0), V_FPR.get('L', 0)),
             '{}|{}|{}'.format(V_AOD.get('W', 0), V_AOD.get('T', 0), V_AOD.get('L', 0)),
             '{}|{}|{}'.format(V_SPD.get('W', 0), V_SPD.get('T', 0), V_SPD.get('L', 0)),
             '{}|{}|{}'.format(V_DIR.get('W', 0), V_DIR.get('T', 0), V_DIR.get('L', 0)),
             '{}|{}|{}'.format(V_FPR_diff.get('W', 0), V_FPR_diff.get('T', 0), V_FPR_diff.get('L', 0)), SUM_W, SUM_T,
             SUM_L])

data_file.close()

