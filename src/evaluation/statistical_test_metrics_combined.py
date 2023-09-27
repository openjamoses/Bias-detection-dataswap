import csv

import pandas as pd
import os
import numpy as np
from scipy import stats
from scipy.stats import ranksums, pearsonr


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
    for f, v in data_before.items():
        for key, val in data_before[f].items():
            for key2, val2 in val.items():
                # accuracy
                if key in PERFORMANCE:
                    _stats, _pvalue = ranksums(data_after[f][key][key2], val2)
                    _delta, _ = cliffDelta(data_after[f][key][key2], val2)
                    rank_val = ranking(_pvalue, _delta)

                    # print(key, key2, _stats, _pvalue, _delta, rank_val)
                else:
                    _stats, _pvalue = ranksums(val2, data_after[f][key][key2])
                    _delta, _ = cliffDelta(val2, data_after[f][key][key2])
                    rank_val = ranking(_pvalue, _delta)
                    # print(key, key2, _stats, _pvalue, _delta, rank_val)
                if key in Rank_dict.keys():
                    if key2 in Rank_dict[key].keys():
                        if rank_val in Rank_dict[key][key2].keys():
                            Rank_dict[key][key2][rank_val] += 1
                        else:
                            Rank_dict[key][key2][rank_val] = 1
                    else:
                        Rank_dict[key][key2] = {}
                        Rank_dict[key][key2][rank_val] = 1
                else:
                    Rank_dict[key] = {}
                    Rank_dict[key][key2] = {}
                    Rank_dict[key][key2][rank_val] = 1
    return Rank_dict
# todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
path_original = '../../dataset-original/'
# path_filtered = '../dataset-filtered/'
# if __name__ == '__main__':
# df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
# data_name = 'adult'
# data_name = 'clevelan_heart'
# data_name = 'clevelan_heart-new'
#data_name = 'Student-new2'
#data_name = 'Student-new'
#data_name = 'clevelan_heart-clean'
#data_name = 'bank-clean'
data_name = 'compas-clean'
#data_name = 'Student-clean'
# data_name = 'compas'
alpha = 0.3
naming = data_name  # '{}-{}_'.format(data_name, alpha)  # _35_threshold
path_output_ = '/Volumes/Cisco/Summer2023/Fairness/Revision/experiments3/{}/'.format(data_name)
#df_lrtd = pd.read_csv(path_original + naming + '/' + str(alpha) + '/' + 'Baseline_LRTD_{}.csv'.format(naming))
BASELINES = ['LRTD', 'fairSMOTE', 'Reweighing', 'DIR']
path_output_ = '/Volumes/Cisco/Summer2023/Fairness/Revision/experiments3/{}/'.format(data_name)
# path_output2 = '/Volumes/Cisco/Fall2022/Fairness/Analysis/Ranking/Shap/'
if not os.path.exists(path_output_):
    os.makedirs(path_output_)
if not os.path.exists(path_output_ + '/Ranking'):
    os.makedirs(path_output_ + '/Ranking')
path_output = path_output_ + '/Ranking/'

data_file = open(path_output + 'statistical-tests-metrics-combined-{}.csv'.format(data_name),
                      mode='w', newline='',
                      encoding='utf-8')
data_writer = csv.writer(data_file)
data_writer.writerow(['Baseline', 'Cases', 'Win-P', 'Tied-P', 'Loss-P', '', 'Win-F', 'Tied-F', 'Loss-F'])
data_performace_metrics = {}
data_fairness_metrics = {}
for basline_ in BASELINES:
    print('Baseline: ', basline_)
    df_data = pd.read_csv(
        path_original + naming + '/' + str(alpha) + '/' + 'Baseline_fairSMOTE_{}.csv'.format(naming))

    #group_data_fairsmote = df_fairsmothe  # df_fairsmothe.groupby(['Feature', 'Droped_Attrib'])['acc_', 'acc_after', 'FPR_diff', 'FPR_diff_after',
    # 'SPD', 'SPD_after', 'DIR','DIR_after','AOD','AOD_after', 'FPR', 'FPR_after', 'SP', 'SP_after', 'Pre', 'Pre_after', 'Re', 'Re_after', 'F1', 'F1_after'].mean().reset_index()

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

    ACC, PRE, RE, F1, FPR, AOD, SPD, DIR, FPR_diff = 'ACC', 'PRE', 'RE', 'F1', 'FPR', 'AOD', 'SPD', 'DIR', 'FPR_diff'




    for i in range(len(Feature_smote)):
        # todo: accuracy
        feature = Feature_smote[i]
        droped_attrib = Droped_Attrib[i]

        if feature in data_performace_metrics.keys():
            if not droped_attrib in data_performace_metrics[feature].keys():
                data_performace_metrics[feature][droped_attrib] = [acc_smote[i], #acc_after_smote[i],
                                                                   Pre_smote[i],
                                                               #Pre_after_smote[i],
                                                                   Re_smote[i], #Re_after_smote[i],
                                                               F1_smote[i], #F1_after_smote[i]
                                                                ]
            else:
                #print(data_performace_metrics.keys())
                #data_performace_metrics[feature][droped_attrib] = [acc_smote[i], acc_after_smote[i],Pre_smote[i], Pre_after_smote[i], Re_smote[i], Re_after_smote[i], F1_smote[i], F1_after_smote[i]]
                data_performace_metrics[feature][droped_attrib].append(acc_smote[i])
                #data_performace_metrics[feature][droped_attrib].append(acc_after_smote[i])
                data_performace_metrics[feature][droped_attrib].append(Pre_smote[i])
                #data_performace_metrics[feature][droped_attrib].append(Pre_after_smote[i])
                data_performace_metrics[feature][droped_attrib].append(Re_smote[i])
                #data_performace_metrics[feature][droped_attrib].append(Re_after_smote[i])
                data_performace_metrics[feature][droped_attrib].append(F1_smote[i])
                #data_performace_metrics[feature][droped_attrib].append(F1_after_smote[i])
        else:
            data_performace_metrics[feature] = {}
            data_performace_metrics[feature][droped_attrib] = [acc_smote[i], #acc_after_smote[i],
                                                               Pre_smote[i],
                                                               #Pre_after_smote[i],
                                                               Re_smote[i], #Re_after_smote[i],
                                                               F1_smote[i], #F1_after_smote[i]
                                                               ]

        if feature in data_fairness_metrics.keys():

            if not droped_attrib in data_fairness_metrics[feature].keys():
                data_fairness_metrics[feature][droped_attrib] = [FPR_smote[i], #FPR_after_smote[i],
                                                                 AOD_smote[i], #AOD_after_smote[i],
                                                                 SPD_smote[i], #SPD_after_smote[i],
                                                                 DIR_smote[i], #DIR_after_smote[i],
                                                                 FPR_diff_smote[i],
                                                                 #FPR_diff_after_smote[i]
                                                                 ]
            else:
                #data_fairness_metrics[feature][droped_attrib] = [FPR_smote[i], FPR_after_smote[i],AOD_smote[i], AOD_after_smote[i], SPD_smote[i], SPD_after_smote[i], DIR_smote[i], DIR_after_smote[i], FPR_diff_smote[i], FPR_diff_after_smote[i]]
                data_fairness_metrics[feature][droped_attrib].append(FPR_smote[i])
                #data_fairness_metrics[feature][droped_attrib].append(FPR_after_smote[i])
                data_fairness_metrics[feature][droped_attrib].append(AOD_smote[i])
                #data_fairness_metrics[feature][droped_attrib].append(AOD_after_smote[i])
                data_fairness_metrics[feature][droped_attrib].append(SPD_smote[i])
                #data_fairness_metrics[feature][droped_attrib].append(SPD_after_smote[i])
                data_fairness_metrics[feature][droped_attrib].append(DIR_smote[i])
                #data_fairness_metrics[feature][droped_attrib].append(DIR_after_smote[i])
                data_fairness_metrics[feature][droped_attrib].append(FPR_diff_smote[i])
                #data_fairness_metrics[feature][droped_attrib].append(FPR_diff_after_smote[i])
        else:
            data_fairness_metrics[feature] = {}
            data_fairness_metrics[feature][droped_attrib] = [FPR_smote[i], FPR_after_smote[i],AOD_smote[i], AOD_after_smote[i], SPD_smote[i], SPD_after_smote[i], DIR_smote[i], DIR_after_smote[i], FPR_diff_smote[i], FPR_diff_after_smote[i]]


new_size = 150
Rank_dict = {}
Rank_dict_perf = {}
for key, val in data_fairness_metrics.items():
    val_before = data_fairness_metrics[key]['None']
    val_before_perf = data_performace_metrics[key]['None']
    #val_before = np.resize(val_before, new_size)
    for key2, val2 in val.items():
        if key != 'None':
            val_after = val2
            val_after_perf = val2

            _stats, _pvalue = ranksums(val_before, val_after)
            _delta, _ = cliffDelta(val_before, val_after)
            rank_val = ranking(_pvalue, _delta)
            print(key,  _stats, _pvalue, _delta, rank_val)
            if key2 in Rank_dict.keys():
                if rank_val in Rank_dict[key2].keys():
                    Rank_dict[key2][rank_val] += 1
                else:
                    Rank_dict[key2][rank_val] = 1
            else:
                Rank_dict[key2] = {}
                Rank_dict[key2][rank_val] = 1

            _stats_perf, _pvalue_perf = ranksums(val_after_perf, val_before_perf)
            _delta_perf, _ = cliffDelta(val_after_perf, val_before_perf)
            rank_val_perf = ranking(_pvalue_perf, _delta_perf)

            if key2 in Rank_dict_perf.keys():
                if rank_val_perf in Rank_dict_perf[key2].keys():
                    Rank_dict_perf[key2][rank_val_perf] += 1
                else:
                    Rank_dict_perf[key2][rank_val_perf] = 1
            else:
                Rank_dict_perf[key2] = {}
                Rank_dict_perf[key2][rank_val_perf] = 1
for key, val in Rank_dict.items():
    val2 = Rank_dict_perf[key]
    data_writer.writerow([basline_, key,val2.get('W', 0), val2.get('T', 0), val2.get('L', 0),'', val.get('W', 0), val.get('T', 0), val.get('L', 0)])
    #data_writer.writerow([basline_, 'Performance', key, val2.get('W', 0), val2.get('T', 0), val2.get('L', 0)])
data_file.close()
