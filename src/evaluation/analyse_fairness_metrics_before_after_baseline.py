import csv

import pandas as pd
import os
import numpy as np
from scipy.stats import ranksums


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

#todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
path_original = '../dataset-original/'
#path_filtered = '../dataset-filtered/'
#if __name__ == '__main__':
#df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')
#data_name = 'adult'
#data_name = 'clevelan_heart'
#data_name = 'clevelan_heart-clean'
#data_name = 'Student-new'
#data_name = 'compas-new2'
#data_name = 'bank-clean'
#data_name = 'compas'
data_name = 'compas-clean'
#data_name = 'bank-clean'
alpha = 0.3
naming = data_name #'{}-{}_'.format(data_name, alpha)  # _35_threshold
path_output_ = '/Volumes/Cisco/Summer2023/Fairness/Revision/experiments2/{}/'.format(data_name)

BASELINES = ['LRTD', 'fairSMOTE', 'Reweighing', 'DIR']
if not os.path.exists(path_output_):
    os.makedirs(path_output_)
if not os.path.exists(path_output_+str(alpha)):
    os.makedirs(path_output_+str(alpha))

path_output = path_output_+str(alpha)

data_file_fairness = open(path_output + '/Fairnmess-metrics-Baseline-after-{}.csv'.format(naming), mode='w', newline='',
                          encoding='utf-8')
data_writer_fairness = csv.writer(data_file_fairness)
data_writer_fairness.writerow(['Features', 'Droped_Attrib', 'Baseline', 'FPRD', 'FPRD_after', 'FPRD_diff',
                               'SPD', 'SPD_after', 'SPD_Diff', 'DIR', 'DIR_after', 'DIR_diff', 'AOD', 'AOD_after',
                               'AOD_diff', 'FPR', 'FPR_after', 'FPR_diff', 'SP', 'SP_after', 'SP_diff', 'acc_',
                               'acc_after', 'acc_diff', 'Pre', 'Pre_after', 'Pre_diff', 'Re', 'Re_after', 'Re_diff',
                               'F1', 'F1_after', 'F1_diff', 'Rank1', 'Rank2', 'Ranking', 'Ranking2_after',
                               'Ranking_before', 'p_value', 'delta'])

for baseline_ in BASELINES:
    print('Baseline: ', baseline_)
    df_data = pd.read_csv(
        path_original + naming + '/' + str(alpha) + '/' + 'Baseline_{}_{}.csv'.format(baseline_, naming))

    group_data = df_data.groupby(['Feature', 'Droped_Attrib'])['acc_', 'acc_after', 'FPR_diff', 'FPR_diff_after',
    'SPD', 'SPD_after', 'DIR','DIR_after','AOD','AOD_after', 'FPR', 'FPR_after', 'SP', 'SP_after', 'Pre', 'Pre_after', 'Re', 'Re_after', 'F1', 'F1_after'].mean().reset_index()

    Feature_smote = group_data.Feature.values.tolist()
    Droped_Attrib = group_data.Droped_Attrib.values.tolist()
    acc_smote = group_data.acc_.values.tolist()
    acc_after_smote = group_data.acc_after.values.tolist()
    FPR_diff_smote = group_data.FPR_diff.values.tolist()
    FPR_diff_after_smote = group_data.FPR_diff_after.values.tolist()
    SPD_smote = group_data.SPD.values.tolist()
    SPD_after_smote = group_data.SPD_after.values.tolist()
    DIR_smote = group_data.DIR.values.tolist()
    DIR_after_smote = group_data.DIR_after.values.tolist()
    AOD_smote = group_data.AOD.values.tolist()
    AOD_after_smote = group_data.AOD_after.values.tolist()
    FPR_smote = group_data.FPR.values.tolist()
    FPR_after_smote = group_data.FPR_after.values.tolist()
    SP_smote = group_data.SP.values.tolist()
    SP_after_smote = group_data.SP_after.values.tolist()
    Pre_smote = group_data.Pre.values.tolist()
    Pre_after_smote = group_data.Pre_after.values.tolist()
    Re_smote = group_data.Re.values.tolist()
    Re_after_smote = group_data.Re_after.values.tolist()

    F1_smote = group_data.F1.values.tolist()
    F1_after_smote = group_data.F1_after.values.tolist()

    dp = 4
    for i in range(len(Feature_smote)):
        ranking_1 = acc_after_smote[i]*Pre_after_smote[i]*Re_after_smote[i]*F1_after_smote[i]
        ranking_11 = acc_after_smote[i]+Pre_after_smote[i]+Re_after_smote[i]+F1_after_smote[i]
        ranking_01 = acc_smote[i] + Pre_smote[i] + Re_smote[i] + F1_smote[i]
        fpr, fpr_diff, spd, aod, dir = FPR_after_smote[i], FPR_diff_after_smote[i], SPD_after_smote[i], AOD_after_smote[i], DIR_after_smote[i]
        most_min = np.min([fpr, fpr_diff, spd, aod, dir])
        most_min_ = np.min([FPR_smote[i], FPR_diff_smote[i], SPD_smote[i], AOD_smote[i], DIR_smote[i]])
        ranking_22 = fpr+abs(most_min) + fpr_diff+abs(most_min) + spd+abs(most_min) + aod+abs(most_min) + dir+abs(most_min)
        ranking_02 = FPR_smote[i]+abs(most_min_)+ FPR_diff_smote[i]+abs(most_min_)+ SPD_smote[i]+abs(most_min_)+ AOD_smote[i]+abs(most_min_)+ DIR_smote[i]+abs(most_min_)
        #if most_min <= 0:
        fpr, fpr_diff, spd, aod, dir = fpr+ abs(most_min)+0.1, fpr_diff+ abs(most_min)+0.1, spd+ abs(most_min)+0.1, aod+ abs(most_min)+0.1, dir+ abs(most_min)+0.1
        #else:
        #    fpr, fpr_diff, spd, aod, dir = fpr+0.1, fpr_diff+0.1, spd+0.1, aod+0.1, dir+0.1
        ranking_2 = fpr*fpr_diff*spd*aod*dir  # abs(FPR_after[i]) * abs(FPR_diff_after[i]) * abs(SPD_after[i]) * abs(AOD_after[i]) * abs(DIR_after[i])
        #print(ranking_1)
        #ranking = '-'
        #if flag2 > 0:

        dist_default = [FPR_diff_smote[i], SPD_smote[i], DIR_smote[i], AOD_smote[i], acc_smote[i], Pre_smote[i], Re_smote[i], F1_smote[i]]
        dist_after = [FPR_diff_after_smote[i], SPD_after_smote[i], DIR_after_smote[i], AOD_after_smote[i], acc_after_smote[i], Pre_after_smote[i],
                        Re_after_smote[i], F1_after_smote[i]]

        statistic, p_value = ranksums(dist_default, dist_after)
        delta, matrix = cliffDelta(dist_default, dist_after)

        ranking = round(ranking_1 / ranking_2, 2)
        ranking_ = round(ranking_11*100 - ranking_22, 2)
        ranking_before = round(ranking_01 * 100 - ranking_02, 2)
        data_writer_fairness.writerow([Feature_smote[i], Droped_Attrib[i], baseline_, round(FPR_diff_smote[i],dp), round(FPR_diff_after_smote[i],dp), round(FPR_diff_after_smote[i]-FPR_diff_smote[i],dp),
                                       round(SPD_smote[i],dp), round(SPD_after_smote[i],dp), round(SPD_after_smote[i]-SPD_smote[i],dp), round(DIR_smote[i],dp), round(DIR_after_smote[i],dp), round(DIR_after_smote[i]-DIR_smote[i],dp), round(AOD_smote[i],dp), round(AOD_after_smote[i],dp),
                                        round(AOD_after_smote[i]-AOD_smote[i],dp), round(FPR_smote[i],dp), round(FPR_after_smote[i],dp), round(FPR_after_smote[i]-FPR_smote[i],dp), round(SP_smote[i],dp), round(SP_after_smote[i],dp), round(SP_after_smote[i]-SP_smote[i],dp), round(acc_smote[i]*100,2),
                                       round(acc_after_smote[i]*100,2), round(acc_after_smote[i]-acc_smote[i],dp), round(Pre_smote[i]*100,2), round(Pre_after_smote[i]*100,2), round(Pre_after_smote[i]-Pre_smote[i],dp), round(Re_smote[i]*100,2), round(Re_after_smote[i]*100,2),round(Re_after_smote[i]-Re_smote[i],dp),
                                       round(F1_smote[i]*100, 2), round(F1_after_smote[i]*100, 2),
                                       round(F1_after_smote[i] - F1_smote[i], dp), ranking_1, ranking_2, ranking, ranking_, ranking_before,
                                       p_value,delta])

data_file_fairness.close()

