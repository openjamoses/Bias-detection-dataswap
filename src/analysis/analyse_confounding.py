import csv

import pandas as pd
import numpy as np
import os

# todo: group by tutorials: https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
alpha = 0.3
path = '../dataset-original/'
# if __name__ == '__main__':
# df = pd.read_csv(path+'logging/log_KL_Divergence_overral_adult-45.csv')

# data_name = 'clevelan_heart'
# data_name = 'compas'
# data_name = 'Student-{}_'.format(alpha)  # _35_threshold
# data_name = 'german_credit'  # _35_threshold
data_name = 'bank-new'
#data_name = 'Student'
path_output = '/Volumes/Cisco/Summer2023/Fairness/Revision/experiments2/{}/'.format(data_name)

swap_proportion_selected = 0.5
# data_name_str = '{}-{}_'.format(data_name, alpha)
df_global = pd.read_csv(path + '{}/{}/Divergence_improved_2_local_{}.csv'.format(data_name, alpha, data_name))
#df_sharp = pd.read_csv(path + '{}/{}/Shap-values_importance_{}-revision.csv'.format(data_name, alpha, data_name))

if not os.path.exists(path_output):
    os.makedirs(path_output)
if not os.path.exists(path_output + '/Natural-impact'):
    os.makedirs(path_output + '/Natural-impact')

print(len(df_global))
# df_global = df_global.dropna(how='any')  # axis='columns'
# print(df_global.columns.tolist())
# df_global = df_global.dropna(subset=['wasserstein_div'], how=-1)
# df_global = df_global[df_global.wasserstein_div != -1]
# print(len(df_global))
'''Feature = df_global.Feature.values.tolist()
Category = df_global.Category.values.tolist()
swap_proportion = df_global.swap_proportion.values.tolist()
Distortion_hellinger = df_global.Distortion_hellinger.values.tolist()
Distortion_wasserstein = df_global.Distortion_wasserstein.values.tolist()
Distortion_js = df_global.Distortion_js.values.tolist()

hellinger_div = df_global.hellinger_div.values.tolist()
wasserstein_div = df_global.wasserstein_div.values.tolist()
JS_Div = df_global.JS_Div.values.tolist()
Casuality = df_global.Casuality.values.tolist()
Importance = df_global.Importance.values.tolist()
for i in range(len(Feature)):'''

# , 'swap_proportion'

data_file = open(path_output + '/Natural-impact/Divergence_2_columns_local_{}.csv'.format(data_name),
                 mode='w', newline='',
                 encoding='utf-8')
data_writer = csv.writer(data_file)

Feature = df_global.Feature.values.tolist()

group_data = df_global  # df_global.groupby(['Feature', 'Feature2', 'swap_proportion', 'Type'])['P_a_b_over_P_a_not_b', 'P_b_a_over_P_b_not_a', 'hellinger_div', 'wasserstein_div',  'total_variation_div', 'JS_Div', 'Casuality', 'Importance'].mean().reset_index()

# group_data_shap = df_sharp.groupby(['Feature', 'swap_proportion'])['Shap', 'Importance'].mean().reset_index()

print(set(Feature))
data_writer.writerow(
    ['Feature', 'Swap \n Percentage', 'Metric', 'CDI', 'NDI', 'NII', 'Impact_Sum', 'AVERAGE', 'NDI2', 'NII2'])

group_data_Importance = df_global.groupby(['Feature', 'swap_proportion'])['Importance'].mean().reset_index()

data_file2 = open(path_output + '/Natural-impact/Confounding_Others_features_{}-revision.csv'.format(data_name),
                  mode='w', newline='',
                  encoding='utf-8')
data_writer2 = csv.writer(data_file2)
data_writer2.writerow(
    ['Feature', 'Swap \n Percentage', 'Metric', 'CDI', 'NDI', 'NII', 'Impact_Sum', 'AVERAGE', 'NDI2', 'NII2'])
data_file3 = open(path_output + '/Natural-impact/Correlated_features_{}-revision.csv'.format(data_name), mode='w',
                  newline='',
                  encoding='utf-8')

data_writer3 = csv.writer(data_file3)

# data_file_shap = open(path_output + '/Natural-impact/Shap_importance_{}-revision.csv'.format(data_name),
#                       mode='w', newline='',
#                       encoding='utf-8')
# data_writer_shap = csv.writer(data_file_shap)
# data_writer_shap.writerow(['Feature', 'Test \n Percentage', 'Metric', 'Value'])
#
# Feature_shap = group_data_shap.Feature.values.tolist()
# Shap = group_data_shap.Shap.values.tolist()
# # Importance = group_data_shap.Importance.values.tolist()
# swap_proportion = group_data_shap.swap_proportion.values.tolist()
#
# # print(group_data_Importance)
# Feature_imp = group_data_Importance.Feature.values.tolist()
# Importance_imp = group_data_Importance.Importance.values.tolist()
# swap_proportion_imp = group_data_Importance.swap_proportion.values.tolist()
# for i in range(len(Feature_shap)):
#     swap_prop = int(swap_proportion[i] * 100)
#     index2 = Feature_imp.index(Feature_shap[i])
#     swap_prop2 = int(swap_proportion_imp[index2] * 100)
#     data_writer_shap.writerow([Feature_shap[i], '{}%'.format(swap_prop), 'Shap\n Value', Shap[i]])
# data_file_shap.close()

Feature = group_data.Feature.values.tolist()
Feature2 = group_data.Feature2.values.tolist()
swap_proportion = group_data.swap_proportion.values.tolist()
Type = group_data.Type.values.tolist()
hellinger_div = group_data.hellinger_div.values.tolist()
wasserstein_div = group_data.wasserstein_div.values.tolist()
total_variation_div = group_data.total_variation_div.values.tolist()
JS_Div = group_data.JS_Div.values.tolist()
Casuality = group_data.Casuality.values.tolist()

P_a_b_over_P_a_not_b = group_data.P_a_b_over_P_a_not_b.values.tolist()
P_b_a_over_P_b_not_a = group_data.P_b_a_over_P_b_not_a.values.tolist()
print(Feature)
print(Feature2)
# feature_set = set(Feature)
## Direct dictionary
data_direct_hellinger = {}
data_direct_wasserstein = {}
data_direct_total_variation = {}
data_direct_JS = {}
data_direct_Casuality = {}
## Indirect dictionary
data_indirect_hellinger = {}
data_indirect_wasserstein = {}
data_indirect_total_variation = {}
data_indirect_JS = {}
data_indirect_Casuality = {}

data_P_a_b_over_P_a_not_b = {}
data_P_b_a_over_P_b_not_a = {}

for i in range(len(Feature)):
    # print(Feature[i], Feature2[i])
    if Feature[i] == Feature2[i]:
        type_ = 'Direct'
        # print(type_, Feature[i], Feature2[i])
        if Feature[i] in data_direct_hellinger.keys():
            if swap_proportion[i] in data_direct_hellinger[Feature[i]].keys():
                if Feature2[i] in data_direct_hellinger[Feature[i]][swap_proportion[i]].keys():
                    data_direct_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(hellinger_div[i])
                    data_direct_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                        wasserstein_div[i])
                    data_direct_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                        total_variation_div[i])
                    data_direct_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(JS_Div[i])
                    data_direct_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(Casuality[i])
                else:
                    data_direct_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [hellinger_div[i]]

                    data_direct_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [wasserstein_div[i]]
                    data_direct_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        [total_variation_div[i]]

                    data_direct_JS[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [JS_Div[i]]

                    data_direct_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_direct_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [Casuality[i]]
            else:
                data_direct_hellinger[Feature[i]][swap_proportion[i]] = {}
                data_direct_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [hellinger_div[i]]

                data_direct_wasserstein[Feature[i]][swap_proportion[i]] = {}
                data_direct_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [wasserstein_div[i]]

                data_direct_total_variation[Feature[i]][swap_proportion[i]] = {}
                data_direct_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    [total_variation_div[i]]

                data_direct_JS[Feature[i]][swap_proportion[i]] = {}
                data_direct_JS[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [JS_Div[i]]

                data_direct_Casuality[Feature[i]][swap_proportion[i]] = {}
                data_direct_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_direct_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [Casuality[i]]

        else:
            data_direct_hellinger[Feature[i]] = {}
            data_direct_hellinger[Feature[i]][swap_proportion[i]] = {}
            data_direct_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [hellinger_div[i]]

            data_direct_wasserstein[Feature[i]] = {}
            data_direct_wasserstein[Feature[i]][swap_proportion[i]] = {}
            data_direct_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [wasserstein_div[i]]

            data_direct_total_variation[Feature[i]] = {}
            data_direct_total_variation[Feature[i]][swap_proportion[i]] = {}
            data_direct_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                [total_variation_div[i]]

            data_direct_JS[Feature[i]] = {}
            data_direct_JS[Feature[i]][swap_proportion[i]] = {}
            data_direct_JS[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [JS_Div[i]]

            data_direct_Casuality[Feature[i]] = {}
            data_direct_Casuality[Feature[i]][swap_proportion[i]] = {}
            data_direct_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_direct_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [Casuality[i]]

    else:
        type_ = 'Indirect'
        # TODO: Indireect casual effect starts here....
        if Feature[i] in data_indirect_hellinger.keys():
            if swap_proportion[i] in data_indirect_hellinger[Feature[i]].keys():
                if Feature2[i] in data_indirect_hellinger[Feature[i]][swap_proportion[i]].keys():
                    if Type[i] in data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]].keys():
                        data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                            hellinger_div[i])
                        data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                            wasserstein_div[i])
                        data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                            total_variation_div[i])
                        data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(JS_Div[i])
                        data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                            Casuality[i])

                        data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                            P_a_b_over_P_a_not_b[i])
                        data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]].append(
                            P_b_a_over_P_b_not_a[i])
                    else:
                        data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [
                            hellinger_div[i]]
                        data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [
                            wasserstein_div[i]]
                        data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            [total_variation_div[i]]
                        data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [JS_Div[i]]
                        data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [Casuality[i]]
                        data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            [P_a_b_over_P_a_not_b[i]]
                        data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                            [P_b_a_over_P_b_not_a[i]]


                else:
                    data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [hellinger_div[i]]

                    data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [
                        wasserstein_div[i]]

                    data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        [total_variation_div[i]]

                    data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [JS_Div[i]]

                    data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [Casuality[i]]

                    data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        [P_a_b_over_P_a_not_b[i]]

                    data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                    data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                        [P_b_a_over_P_b_not_a[i]]
            else:
                data_indirect_hellinger[Feature[i]][swap_proportion[i]] = {}
                data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [hellinger_div[i]]

                data_indirect_wasserstein[Feature[i]][swap_proportion[i]] = {}
                data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [wasserstein_div[i]]

                data_indirect_total_variation[Feature[i]][swap_proportion[i]] = {}
                data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    [total_variation_div[i]]

                data_indirect_JS[Feature[i]][swap_proportion[i]] = {}
                data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [JS_Div[i]]

                data_indirect_Casuality[Feature[i]][swap_proportion[i]] = {}
                data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [Casuality[i]]

                data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]] = {}
                data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [P_a_b_over_P_a_not_b[
                                                                                                       i]]

                data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]] = {}
                data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
                data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                    [P_b_a_over_P_b_not_a[i]]

        else:
            data_indirect_hellinger[Feature[i]] = {}
            data_indirect_hellinger[Feature[i]][swap_proportion[i]] = {}
            data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_hellinger[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [hellinger_div[i]]

            data_indirect_wasserstein[Feature[i]] = {}
            data_indirect_wasserstein[Feature[i]][swap_proportion[i]] = {}
            data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_wasserstein[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [wasserstein_div[i]]

            data_indirect_total_variation[Feature[i]] = {}
            data_indirect_total_variation[Feature[i]][swap_proportion[i]] = {}
            data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_total_variation[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                [total_variation_div[i]]

            data_indirect_JS[Feature[i]] = {}
            data_indirect_JS[Feature[i]][swap_proportion[i]] = {}
            data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_JS[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [JS_Div[i]]

            data_indirect_Casuality[Feature[i]] = {}
            data_indirect_Casuality[Feature[i]][swap_proportion[i]] = {}
            data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_indirect_Casuality[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [Casuality[i]]

            data_P_a_b_over_P_a_not_b[Feature[i]] = {}
            data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]] = {}
            data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_P_a_b_over_P_a_not_b[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = [P_a_b_over_P_a_not_b[
                                                                                                   i]]

            data_P_b_a_over_P_b_not_a[Feature[i]] = {}
            data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]] = {}
            data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]] = {}
            data_P_b_a_over_P_b_not_a[Feature[i]][swap_proportion[i]][Feature2[i]][Type[i]] = \
                [P_b_a_over_P_b_not_a[i]]

row = ['Swap \n Percentage', 'Measurement', 'Feature']
# feature_set = set(['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'])
# feature_list = ['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'] #list(feature_set)
# feature_list = ['sex', 'age', 'thalach', 'ca', 'thal', 'exang', 'cp', 'trestbps', 'restecg', 'fbs', 'oldpeak',
#                'chol']  # list(feature_set)
# feature_list = ['race', 'sex', 'age', 'hours-per-week', 'capital-gain', 'capital-loss'] #list(feature_set)
# feature_list = ['sex','age','health','Pstatus','nursery','Medu', 'Fjob', 'schoolsup', 'absences',  'activities', 'higher', 'traveltime',  'paid', 'guardian',  'Walc', 'freetime', 'famsup',  'romantic', 'studytime', 'goout', 'reason',  'famrel', 'internet']
# feature_list = [ 'Sex', 'Age','Job', 'Saving', 'Checking', 'Credit','Housing', 'Purpose']
# feature_list = ['race','sex', 'age', 'c_charge_degree', 'priors_count']
# feature_list = ['age', 'education', 'job', 'loan', 'balance', 'housing', 'duration', 'campaign', 'default']
data_features_dict = {}
for i in range(len(Feature)):
    if Feature[i] in data_features_dict.keys():
        data_features_dict[Feature[i]].append(Feature2[i])
    else:
        data_features_dict[Feature[i]] = [Feature2[i]]
list_features_keys = data_features_dict.keys()
feature_list2 = set(Feature2)
for feature_ in feature_list2:
    row.append(feature_)
row.append('')
row.append('CDI')
row.append('SUM')
row.append('AVERAGE')
row.append('SUM_COMBINED')
row.append('RAW_NDI')
row.append('RAW_NII')
row.append('RAW_NDI_NII')
data_writer3.writerow(row)

data_probability_mapping_CDI = {}
data_probability_mapping_NDI = {}
data_probability_mapping_NII = {}
data_probability_mapping_NDI_NII = {}

data_probability_NDI_a_b = {}
data_probability_NII_a_b = {}

# print(data_direct_hellinger)
for feature_ in list_features_keys:
    if feature_ in data_direct_hellinger.keys():
        val = data_direct_hellinger.get(feature_)
        # print(feature_, val)
        for swap_, val2_ in val.items():
            type_ = 'CDI'
            swap_prop = int(swap_)  # int(swap_ * 100)
            direct_effect_hellinger = np.mean(val2_[feature_][type_])
            direct_effect_wasserstein = np.mean(data_direct_wasserstein[feature_][swap_][feature_][type_])
            direct_effect_total_variation = np.mean(data_direct_total_variation[feature_][swap_][feature_][type_])
            direct_effect_JS = np.mean(data_direct_JS[feature_][swap_][feature_][type_])
            direct_effect_Casuality = np.mean(data_direct_Casuality[feature_][swap_][feature_][type_])
            if not feature_ in data_probability_mapping_CDI.keys():
                data_probability_mapping_CDI[feature_] = {}
            if not swap_prop in data_probability_mapping_CDI[feature_].keys():
                data_probability_mapping_CDI[feature_][swap_prop] = {}
            data_probability_mapping_CDI[feature_][swap_prop]['hellinger'] = direct_effect_hellinger
            data_probability_mapping_CDI[feature_][swap_prop]['wasserstein'] = direct_effect_wasserstein
            data_probability_mapping_CDI[feature_][swap_prop]['total_variation'] = direct_effect_total_variation
            data_probability_mapping_CDI[feature_][swap_prop]['JS'] = direct_effect_JS
            data_probability_mapping_CDI[feature_][swap_prop]['Casuality'] = direct_effect_Casuality
            # print(swap_)
            if feature_ in data_indirect_hellinger.keys():
                for feature_2 in data_features_dict[feature_]:
                    # for feature_2, val2 in data_indirect_hellinger[feature_][swap_].items():
                    if feature_2 in data_indirect_hellinger[feature_][swap_].keys():
                        val2 = data_indirect_hellinger[feature_][swap_][feature_2]
                        # todo: direct impact results 1
                        type_NDI = 'NDI'
                        type_NII = 'NII'
                        # todo: NDI results 1
                        # print(data_indirect_hellinger)
                        NDI_effect_hellinger = np.mean(data_indirect_hellinger[feature_][swap_][feature_2][type_NDI])
                        NDI_effect_wasserstein = np.mean(
                            data_indirect_wasserstein[feature_][swap_][feature_2][type_NDI])
                        NDI_effect_total_variation = np.mean(
                            data_indirect_total_variation[feature_][swap_][feature_2][type_NDI])
                        # print(data_indirect_JS[feature_][swap_], data_indirect_JS[feature_][swap_][keys_[1]])
                        NDI_effect_JS = np.mean(data_indirect_JS[feature_][swap_][feature_2][type_NDI])
                        # print(data_indirect_Casuality[feature_][swap_], data_indirect_Casuality[feature_][swap_][keys_[1]])
                        NDI_effect_Casuality = np.mean(data_indirect_Casuality[feature_][swap_][feature_2][type_NDI])

                        val_NDI_P_a_b_over_P_a_not_b = np.mean(
                            data_P_a_b_over_P_a_not_b[feature_][swap_][feature_2][type_NDI])
                        val_NDI_P_b_a_over_P_b_not_a = np.mean(
                            data_P_b_a_over_P_b_not_a[feature_][swap_][feature_2][type_NDI])

                        val_NII_P_a_b_over_P_a_not_b = np.mean(
                            data_P_a_b_over_P_a_not_b[feature_][swap_][feature_2][type_NII])
                        val_NII_P_b_a_over_P_b_not_a = np.mean(
                            data_P_b_a_over_P_b_not_a[feature_][swap_][feature_2][type_NII])

                        # todo: NII results 1
                        NII_effect_hellinger = np.mean(data_indirect_hellinger[feature_][swap_][feature_2][type_NII])
                        NII_effect_wasserstein = np.mean(
                            data_indirect_wasserstein[feature_][swap_][feature_2][type_NII])
                        NII_effect_total_variation = np.mean(
                            data_indirect_total_variation[feature_][swap_][feature_2][type_NII])
                        # print(data_indirect_JS[feature_][swap_], data_indirect_JS[feature_][swap_][keys_[1]])
                        NII_effect_JS = np.mean(data_indirect_JS[feature_][swap_][feature_2][type_NII])
                        # print(data_indirect_Casuality[feature_][swap_], data_indirect_Casuality[feature_][swap_][keys_[1]])
                        NII_effect_Casuality = np.mean(data_indirect_Casuality[feature_][swap_][feature_2][type_NII])

                        if not feature_ + '/' + feature_2 in data_probability_mapping_NDI.keys():
                            data_probability_mapping_NDI[feature_ + '/' + feature_2] = {}
                        if not swap_prop in data_probability_mapping_NDI[feature_ + '/' + feature_2].keys():
                            data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop] = {}
                        data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop][
                            'hellinger'] = NDI_effect_hellinger
                        data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop][
                            'wasserstein'] = NDI_effect_wasserstein
                        data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop][
                            'total_variation'] = NDI_effect_total_variation
                        data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop]['JS'] = NDI_effect_JS
                        data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop][
                            'Casuality'] = NDI_effect_Casuality

                        if not feature_ + '/' + feature_2 in data_probability_mapping_NII.keys():
                            data_probability_mapping_NII[feature_ + '/' + feature_2] = {}
                        if not swap_prop in data_probability_mapping_NII[feature_ + '/' + feature_2].keys():
                            data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop] = {}

                        # data_probability_mapping_NII[feature_+'/'+feature_2] = {}
                        # data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop] = {}
                        data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop][
                            'hellinger'] = NII_effect_hellinger
                        data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop][
                            'wasserstein'] = NII_effect_wasserstein
                        data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop][
                            'total_variation'] = NII_effect_total_variation
                        data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['JS'] = NII_effect_JS
                        data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop][
                            'Casuality'] = NII_effect_Casuality

                        if not feature_ + '/' + feature_2 in data_probability_NII_a_b.keys():
                            data_probability_NII_a_b[feature_ + '/' + feature_2] = {}
                        if not swap_prop in data_probability_NII_a_b[feature_ + '/' + feature_2].keys():
                            data_probability_NII_a_b[feature_ + '/' + feature_2][swap_prop] = {}

                        data_probability_NII_a_b[feature_ + '/' + feature_2][
                            swap_prop] = val_NII_P_a_b_over_P_a_not_b - val_NII_P_b_a_over_P_b_not_a

                        ##todo: Combining the two
                        '''if not feature_ + '/' + feature_2 in data_probability_mapping_NDI_NII.keys():
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2] = {}
                        if not swap_prop in data_probability_mapping_NDI_NII[feature_ + '/' + feature_2].keys():
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop] = {}
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['hellinger'] = NDI_effect_hellinger + NII_effect_hellinger
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'wasserstein'] = NDI_effect_wasserstein+NII_effect_wasserstein
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['cramers_v'] = NDI_effect_cramers_v+NII_effect_cramers_v
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'total_variation'] = NDI_effect_total_variation+NII_effect_total_variation
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['JS'] = NDI_effect_JS+NII_effect_JS
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'] = NDI_effect_Casuality+NII_effect_Casuality
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'effect_distance'] = NDI_effect_distance+NII_effect_distance
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'distance_contrained'] = NDI_effect_distance_contrained+NII_effect_distance_contrained'''
                        if not feature_ + '/' + feature_2 in data_probability_mapping_NDI_NII.keys():
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2] = {}
                        if not swap_prop in data_probability_mapping_NDI_NII[feature_ + '/' + feature_2].keys():
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop] = {}
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'hellinger'] = NDI_effect_hellinger  # + NII_effect_hellinger
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'wasserstein'] = NDI_effect_wasserstein  # +NII_effect_wasserstein
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'total_variation'] = NDI_effect_total_variation + NII_effect_total_variation
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'JS'] = NDI_effect_JS  # +NII_effect_JS
                        data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                            'Casuality'] = NDI_effect_Casuality  # +NII_effect_Casuality

swap_selected = [10, 30, 50, 70]  #
for feature_ in list_features_keys:
    if feature_ in data_probability_mapping_CDI.keys():
        val = data_probability_mapping_CDI.get(feature_)
        # print(val.keys())
        for swap_prop, val2_ in val.items():
            if swap_prop in swap_selected:
                row_hellinger = ['{}%'.format(swap_prop), 'Hellinger\n distance', feature_]
                row_wasserstein = ['{}%'.format(swap_prop), 'Wasserstein\n distance', feature_]
                row_total_variation = ['{}%'.format(swap_prop), 'Total variation\n distance', feature_]
                row_js = ['{}%'.format(swap_prop), 'Jensen-Shannon\n divergence', feature_]
                row_Casuality = ['{}%'.format(swap_prop), 'Propensity\n score', feature_]

                value_hellinger_NDI = []
                value_wasserstein_NDI = []
                value_total_variation_NDI = []
                value_js_NDI = []
                value_Casuality_NDI = []

                value_hellinger_NII = []
                value_wasserstein_NII = []
                value_total_variation_NII = []
                value_js_NII = []
                value_Casuality_NII = []

                value_hellinger_NDI_NII = []
                value_wasserstein_NDI_NII = []
                value_total_variation_NDI_NII = []
                value_js_NDI_NII = []

                value_Casuality_NDI_NII = []

                # print(val, data_probability_mapping_CDI)

                for feature_2 in feature_list2:
                    if feature_ + '/' + feature_2 in data_probability_mapping_NDI.keys():
                        row_hellinger.append(str(
                            round(data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop]['hellinger'],
                                  4)) + '/' + str(
                            round(data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['hellinger'], 4)))
                        row_wasserstein.append(str(
                            round(data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop]['wasserstein'],
                                  4)) + '/' + str(
                            round(data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['wasserstein'],
                                  4)))
                        row_total_variation.append(str(
                            round(
                                data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop]['total_variation'],
                                4)) + '/' + str(
                            round(
                                data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['total_variation'],
                                4)))
                        row_js.append(str(
                            round(data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop]['JS'],
                                  4)) + '/' + str(
                            round(data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['JS'], 4)))

                        row_Casuality.append(str(
                            round(data_probability_mapping_NDI[feature_ + '/' + feature_2][swap_prop]['Casuality'],
                                  4)) + '/' + str(
                            round(data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'], 4)))
                        ## todo: Natural indirect impact
                        val_h = 0
                        val_w = 0
                        val_c = 0
                        val_t = 0
                        val_s = 0

                        val_cas = 0
                        val_eff = 0
                        val_dCons = 0

                        value_hellinger_NII.append(
                            data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['hellinger'])
                        val_h = data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['hellinger']

                        value_wasserstein_NII.append(
                            data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['wasserstein'])
                        val_w = data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['wasserstein']

                        value_total_variation_NII.append(
                            data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['total_variation'])
                        val_t = data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['total_variation']

                        value_js_NII.append(data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['JS'])
                        val_s = data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['JS']

                        ## Added temporary
                        value_Casuality_NII.append(
                            data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['Casuality'])
                        val_cas = data_probability_mapping_NII[feature_ + '/' + feature_2][swap_prop]['Casuality']

                        ## todo: Combined Natural direct and Indirect impact
                        value_hellinger_NDI_NII.append(
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'hellinger'] + val_h)

                        value_wasserstein_NDI_NII.append(
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'wasserstein'] + val_w)

                        value_total_variation_NDI_NII.append(
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'total_variation'] + val_t)
                        value_js_NDI_NII.append(
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop]['JS'] + val_s)

                        value_Casuality_NDI_NII.append(
                            data_probability_mapping_NDI_NII[feature_ + '/' + feature_2][swap_prop][
                                'Casuality'] + val_cas)

                    else:
                        row_hellinger.append('-')
                        row_wasserstein.append('-')
                        row_total_variation.append('-')
                        row_js.append('-')

                        row_Casuality.append('-')

                hellinger_NDI_mean = 0
                wasserstein_NDI_mean = 0
                cramers_v_NDI_mean = 0
                total_variation_NDI_mean = 0
                JS_NDI_mean = 0

                Casuality_NDI_mean = 0
                effect_distance_NDI_mean = 0
                effect_distance_constained_NDI_mean = 0
                if len(value_hellinger_NDI) > 0:
                    hellinger_NDI_mean = np.sum(value_hellinger_NDI)
                if len(value_wasserstein_NDI) > 0:
                    wasserstein_NDI_mean = np.sum(value_wasserstein_NDI)
                if len(value_total_variation_NDI) > 0:
                    total_variation_NDI_mean = np.sum(value_total_variation_NDI)
                if len(value_js_NDI) > 0:
                    JS_NDI_mean = np.sum(value_js_NDI)

                if len(value_Casuality_NDI) > 0:
                    Casuality_NDI_mean = np.sum(value_Casuality_NDI)
                # todo: natual indirect
                hellinger_NII_mean = 0
                wasserstein_NII_mean = 0
                cramers_v_NII_mean = 0
                total_variation_NII_mean = 0
                JS_NII_mean = 0

                Casuality_NII_mean = 0
                effect_distance_NII_mean = 0
                effect_distance_constained_NII_mean = 0
                if len(value_hellinger_NII) > 0:
                    hellinger_NII_mean = np.sum(value_hellinger_NII)
                if len(value_wasserstein_NII) > 0:
                    wasserstein_NII_mean = np.sum(value_wasserstein_NII)
                if len(value_total_variation_NII) > 0:
                    total_variation_NII_mean = np.sum(value_total_variation_NII)
                if len(value_js_NII) > 0:
                    JS_NII_mean = np.sum(value_js_NII)

                if len(value_Casuality_NII) > 0:
                    Casuality_NII_mean = np.sum(value_Casuality_NII)

                # todo: combined natural direct and indirect impact
                hellinger_NDI_NII_mean = 0
                wasserstein_NDI_NII_mean = 0
                cramers_v_NDI_NII_mean = 0
                total_variation_NDI_NII_mean = 0
                JS_NDI_NII_mean = 0
                if len(value_hellinger_NDI_NII) > 0:
                    hellinger_NDI_NII_mean = np.sum(value_hellinger_NDI_NII)
                if len(value_wasserstein_NDI_NII) > 0:
                    wasserstein_NDI_NII_mean = np.sum(value_wasserstein_NDI_NII)
                if len(value_total_variation_NDI_NII) > 0:
                    total_variation_NDI_NII_mean = np.sum(value_total_variation_NDI_NII)
                if len(value_js_NDI_NII) > 0:
                    JS_NDI_NII_mean = np.sum(value_js_NDI_NII)

                Casuality_NDI_NII_mean = 0
                effect_distance_NDI_NII_mean = 0
                effect_distance_constained_NDI_NII_mean = 0
                if len(value_Casuality_NDI_NII) > 0:
                    Casuality_NDI_NII_mean = np.sum(value_Casuality_NDI_NII)

                row_hellinger.append('')
                row_hellinger.append(data_probability_mapping_CDI[feature_][swap_prop]['hellinger'])
                row_hellinger.append(
                    data_probability_mapping_CDI[feature_][swap_prop]['hellinger'] + hellinger_NDI_mean)
                row_hellinger.append(data_probability_mapping_CDI[feature_][swap_prop]['hellinger'] + (
                        hellinger_NDI_mean + hellinger_NII_mean) / 2)
                # row_hellinger.append((data_probability_mapping_CDI[feature_][swap_prop]['hellinger'] + hellinger_NDI_NII_mean)/2)
                row_hellinger.append(hellinger_NDI_NII_mean)

                row_hellinger.append(value_hellinger_NDI)
                row_hellinger.append(value_hellinger_NII)
                row_hellinger.append(value_hellinger_NDI_NII)
                data_writer3.writerow(row_hellinger)

                row_wasserstein.append('')
                row_wasserstein.append(data_probability_mapping_CDI[feature_][swap_prop]['wasserstein'])
                row_wasserstein.append(
                    data_probability_mapping_CDI[feature_][swap_prop]['wasserstein'] + wasserstein_NDI_mean)
                row_wasserstein.append(data_probability_mapping_CDI[feature_][swap_prop][
                                           'wasserstein'] + (wasserstein_NDI_mean + wasserstein_NII_mean) / 2)

                # row_wasserstein.append(
                #    (data_probability_mapping_CDI[feature_][swap_prop]['wasserstein'] + wasserstein_NDI_NII_mean)/2)

                row_wasserstein.append(wasserstein_NDI_NII_mean)
                row_wasserstein.append(value_wasserstein_NDI)
                row_wasserstein.append(value_wasserstein_NII)
                row_wasserstein.append(value_wasserstein_NDI_NII)
                data_writer3.writerow(row_wasserstein)

                # data_writer3.writerow(row_cramers_v)

                row_total_variation.append('')
                row_total_variation.append(data_probability_mapping_CDI[feature_][swap_prop]['total_variation'])
                row_total_variation.append(
                    data_probability_mapping_CDI[feature_][swap_prop]['total_variation'] + total_variation_NDI_mean)
                row_total_variation.append(data_probability_mapping_CDI[feature_][swap_prop][
                                               'total_variation'] + (
                                                   total_variation_NDI_mean + total_variation_NII_mean) / 2)
                # row_total_variation.append(
                #    (data_probability_mapping_CDI[feature_][swap_prop]['total_variation'] + total_variation_NDI_NII_mean)/2)
                row_total_variation.append(total_variation_NDI_NII_mean)
                row_total_variation.append(value_total_variation_NDI)
                row_total_variation.append(value_total_variation_NII)
                row_total_variation.append(value_total_variation_NDI_NII)
                data_writer3.writerow(row_total_variation)

                row_js.append('')
                row_js.append(data_probability_mapping_CDI[feature_][swap_prop]['JS'])
                row_js.append(
                    data_probability_mapping_CDI[feature_][swap_prop]['JS'] + JS_NDI_mean)
                row_js.append(data_probability_mapping_CDI[feature_][swap_prop][
                                  'JS'] + (JS_NDI_mean + JS_NII_mean) / 2)
                # row_js.append(
                #    (data_probability_mapping_CDI[feature_][swap_prop]['JS']+ JS_NDI_NII_mean)/2)
                row_js.append(JS_NDI_NII_mean)

                row_js.append(value_js_NDI)
                row_js.append(value_js_NII)
                row_js.append(value_js_NDI_NII)

                data_writer3.writerow(row_js)

                ## temporary added
                row_Casuality.append('')
                row_Casuality.append(data_probability_mapping_CDI[feature_][swap_prop]['Casuality'])
                row_Casuality.append(
                    data_probability_mapping_CDI[feature_][swap_prop]['Casuality'] + Casuality_NDI_mean)
                row_Casuality.append(data_probability_mapping_CDI[feature_][swap_prop][
                                         'Casuality'] + (Casuality_NDI_mean + Casuality_NII_mean) / 2)
                row_Casuality.append(
                    (data_probability_mapping_CDI[feature_][swap_prop]['Casuality'] + Casuality_NDI_NII_mean) / 2)
                row_Casuality.append(value_Casuality_NDI)
                row_Casuality.append(value_Casuality_NII)
                row_Casuality.append(value_Casuality_NDI_NII)

                # data_writer3.writerow(row_Casuality)

                data_writer.writerow(
                    [feature_, '{}%'.format(swap_prop), 'Hellinger\n distance(+)',
                     data_probability_mapping_CDI[feature_][swap_prop]['hellinger'],
                     hellinger_NDI_mean,
                     hellinger_NII_mean,
                     data_probability_mapping_CDI[feature_][swap_prop][
                         'hellinger'] + hellinger_NDI_mean + hellinger_NII_mean,
                     (data_probability_mapping_CDI[feature_][swap_prop][
                          'hellinger'] + hellinger_NDI_mean + hellinger_NII_mean) / 3])

                data_writer.writerow(
                    [feature_, '{}%'.format(swap_prop), 'Wasserstein\n distance(+)',
                     data_probability_mapping_CDI[feature_][swap_prop]['wasserstein'],
                     hellinger_NDI_mean,
                     hellinger_NII_mean,
                     data_probability_mapping_CDI[feature_][swap_prop][
                         'wasserstein'] + wasserstein_NDI_mean * wasserstein_NII_mean,
                     (data_probability_mapping_CDI[feature_][swap_prop][
                          'wasserstein'] + wasserstein_NDI_mean + wasserstein_NII_mean) / 3])

                data_writer.writerow(
                    [feature_, '{}%'.format(swap_prop), 'Total variation\n distance(+)',
                     data_probability_mapping_CDI[feature_][swap_prop]['total_variation'],
                     total_variation_NDI_mean,
                     total_variation_NII_mean,
                     data_probability_mapping_CDI[feature_][swap_prop][
                         'total_variation'] + total_variation_NDI_mean * total_variation_NII_mean,
                     (data_probability_mapping_CDI[feature_][swap_prop][
                          'total_variation'] + total_variation_NDI_mean + total_variation_NII_mean) / 3])

                data_writer.writerow(
                    [feature_, '{}%'.format(swap_prop), 'Jensen–Shannon\n divergence(+)',
                     data_probability_mapping_CDI[feature_][swap_prop]['JS'],
                     JS_NDI_mean,
                     JS_NII_mean,
                     data_probability_mapping_CDI[feature_][swap_prop]['JS'] + JS_NDI_mean * JS_NII_mean,
                     (data_probability_mapping_CDI[feature_][swap_prop][
                          'JS'] + JS_NDI_mean + JS_NII_mean) / 3])

                ##TODO; Other casual metrics

                '''data_writer2.writerow(
                    [feature_, '{}%'.format(swap_prop), 'Impact\n distance(+)', direct_effect_distance,
                     NDI_effect_distance, NII_effect_distance,
                     direct_effect_distance + NDI_effect_distance + NII_effect_distance,
                     (direct_effect_distance + NDI_effect_distance + NII_effect_distance) / 3])

                data_writer2.writerow(
                    [feature_, '{}%'.format(swap_prop), 'Impact \n Distance Contrained(+)', direct_effect_distance_contrained,
                     NDI_effect_distance_contrained, NII_effect_distance_contrained,
                     direct_effect_distance_contrained + NDI_effect_distance_contrained + NII_effect_distance_contrained,
                     (direct_effect_distance_contrained + NDI_effect_distance_contrained + NII_effect_distance_contrained) / 3])

                data_writer2.writerow(
                    [feature_, '{}%'.format(swap_prop), 'Propensity\n score(+)',
                     direct_effect_Casuality,
                     NDI_effect_Casuality, NII_effect_Casuality,
                     direct_effect_Casuality + NDI_effect_Casuality + NII_effect_Casuality,
                     (direct_effect_Casuality + NDI_effect_Casuality + NII_effect_Casuality) / 3])'''

data_file.close()
data_file2.close()
data_file3.close()

