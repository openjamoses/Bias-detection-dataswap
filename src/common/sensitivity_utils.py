import math

from scipy.stats import gaussian_kde, entropy, ks_2samp
from numpy import concatenate, linspace
import random
import numpy as np
import matplotlib.pyplot as plt

from src.common.utility_functions import data_split

MAX_ENTROPY_ALLOWED = 1e6  # A hack to never deal with inf entropy values that happen when the PDFs don't intersect

def is_categorical(data, feature_index):
    #print(data[0:, feature_index].dtype.name)
    unique_values = np.unique(data[0:, feature_index])
    #return data[0:, feature_index].dtype.name == 'category'
    if len(unique_values) < 4:
        return True
    else:
        return False
def change_category_value(list_category=[0,1], value=0):
    categories = []
    categories.extend(list_category)
    if value in categories:
        categories.remove(value)
    #print(categories, list_category, value)
    if len(categories) > 0:
        return random.choice(categories)
    else:
        return value
def alter_feature_values_categorical_3(x_test, y_test, posible_values=[0,1],feature_index=0, swap_proportion=0.01):
    new_test_data = []
    random_index = [i for i in range(len(y_test))]
    #print('random_index: ', random_index, swap_proportion)
    if round(len(random_index)*swap_proportion, 0) == len(random_index): # and swap_proportion > 0.9:
        _, random_index_ = 0, random_index
    else:
        _, random_index_ = data_split(data=random_index, sample_size=swap_proportion)
    for i in range(len(x_test)):
        if i in random_index_:
            row = []
            for j in range(len(x_test[i])):
                temp_val = x_test[i][j]
                if j == feature_index:
                    temp_val = change_category_value(posible_values, x_test[i][j])
                #if test_data[i][j] == posible_values[0] and j == feature_index:
                #    temp_val = posible_values[1]
                #elif test_data[i][j] == posible_values[1] and j == feature_index:
                #    temp_val = posible_values[0]
                row.append(temp_val)
            new_test_data.append(row)
        else:
            new_test_data.append([x for x in x_test[i]])
    return random_index_, np.array(new_test_data)
def check_feature_value_belong_to(folded_data, value):
    posible_choice = []
    posible_choice2 = []
    for k, vL in folded_data.items():
        #vL = list(vL)
        #print(value, vL)
        if value in vL:
            #print(vL, vL[0], vL[1], value, random.uniform(0.0, 0.02))
            posible_choice2.append(random.choice(vL))
        else:
            posible_choice.append(random.choice(vL))
        #if not value in vL:
        #    posible_choice.extend(vL)
        #else:
        #    posible_choice2.extend(vL)
    #print(posible_choice, folded_data)
    if len(posible_choice)>0:
        return random.choice(posible_choice)
    else:
        return random.choice(posible_choice2)
def alter_2_columns(x_test, y_test, posible_values={},feature_indices=[0,1], swap_proportion=0.01):


    new_test_data_changed_both = []
    new_test_data_changed_feature_only = []
    new_test_data_changed_mediator_only = []
    random_index = [i for i in range(len(y_test))]
    if round(len(random_index) * swap_proportion, 0) == len(random_index):
        _, random_index_ = 0, random_index
    else:
        _, random_index_ = data_split(data=random_index, sample_size=swap_proportion)
    for i in range(len(x_test)):
        if i in random_index_:
            row_changed_both = []
            row_feature_only = []
            row_mediator_only = []
            for j in range(len(x_test[i])):
                temp_val = x_test[i][j]
                #temp_val_original = x_test[i][j]
                temp_val_feature_only = x_test[i][j]
                temp_val_mediator_only = x_test[i][j]
                if j == feature_indices[0]:
                    if is_categorical(x_test, feature_indices[0]):
                        temp_val = change_category_value(posible_values[feature_indices[0]], x_test[i][j])
                    else:
                        temp_val = check_feature_value_belong_to(posible_values[feature_indices[0]], x_test[i][j])
                    temp_val_feature_only = temp_val
                    temp_val_mediator_only = x_test[i][j]

                if j == feature_indices[1]:
                    if is_categorical(x_test, feature_indices[1]):
                        #print(j, posible_values[feature_indices[1]], posible_values)
                        temp_val = change_category_value(posible_values[feature_indices[1]], x_test[i][j])
                    else:
                        temp_val = check_feature_value_belong_to(posible_values[feature_indices[1]], x_test[i][j])
                    temp_val_feature_only = x_test[i][j]
                    temp_val_mediator_only = temp_val
                row_changed_both.append(temp_val)
                row_feature_only.append(temp_val_feature_only)
                row_mediator_only.append(temp_val_mediator_only)
            #if len(row) > 0:
            new_test_data_changed_both.append(row_changed_both)
            new_test_data_changed_feature_only.append(row_feature_only)
            new_test_data_changed_mediator_only.append(row_mediator_only)
        else:
            org_val = [x for x in x_test[i]]
            new_test_data_changed_both.append(org_val)
            new_test_data_changed_feature_only.append(org_val)
            new_test_data_changed_mediator_only.append(org_val)
    #print(new_test_data)
    return random_index_, np.array(new_test_data_changed_feature_only), np.array(new_test_data_changed_mediator_only), np.array(new_test_data_changed_both)


def alter_feature_value_continous_3(x_test, y_test, posible_values={},feature_index=0, swap_proportion=0.01):
    new_test_data = []
    random_index = [i for i in range(len(y_test))]
    if round(len(random_index)*swap_proportion, 0) == len(random_index):
        _, random_index_ = 0, random_index
    else:
        _, random_index_ = data_split(data=random_index, sample_size=swap_proportion)
    #x_sampled_k, y_sampled_k = [], []
    for i in range(len(x_test)):
        if i in random_index_:
            row = []
            for j in range(len(x_test[i])):
                temp_val = x_test[i][j]
                if j == feature_index:
                    temp_val = check_feature_value_belong_to(posible_values, x_test[i][j])
                row.append(temp_val)
            if len(row) > 0:
                new_test_data.append(row)
        else:
            new_test_data.append([x for x in x_test[i]])
    #print(new_test_data)
    return random_index_, np.array(new_test_data)


def run_dl_divergence(p_samples, q_samples, p_label="P", q_label="Q", column_index=0, sub_category=None, show_figure=False, path='../dataset/'):
    n1 = len(p_samples)
    n2 = len(q_samples)
    if n1 == 0 or n2 == 0:
        return 0.0, None,None, None

    # Plot the samples
    #print(p_samples)
    plt.hist(p_samples, density=True, bins=25, color='b', alpha=0.5, label=p_label)
    #plt.hist()
    plt.hist(q_samples, density=True, bins=25, color='g', alpha=0.5, label=q_label)
    plt.legend(loc="upper right")
    #plt.show()

    try:
        # Estimate the PDFs using Gaussian KDE
        pdf1 = gaussian_kde(p_samples)
        pdf2 = gaussian_kde(q_samples)

        # Calculate the interval to be analyzed further
        a = min(min(p_samples), min(q_samples))
        b = max(max(p_samples), max(q_samples))

        # Plot the PDFs
        lin = linspace(a, b, max(n1, n2))
        p = pdf1.pdf(lin)
        q = pdf2.pdf(lin)

        #print('line: ', len(lin))
        #print(p,q, len(p_samples), len(q_samples), len(p), len(q))
        plt.plot(lin, p, color='b', label="Estimated PDF({})".format(p_label))
        plt.plot(lin, q, color='g', label="Estimated PDF({})".format(q_label))
        plt.legend(loc="upper right")
        plt.title('F_{}'.format(column_index))
        #if show_figure:

        if sub_category != None:
            plt.title('F_{}: {}'.format(column_index,sub_category))
            plt.savefig(path+"{}.png".format(sub_category))
        plt.show()

        min_p = min(p_samples)
        max_p = max(p_samples)

        min_q = min(p_samples)
        max_q = max(p_samples)

        DL = np.log(max_q/max_p) + (max_p**2+(min_p-min_q)**2)/(2*max_q**2) -0.5
        print('DL manually: ', DL)
        # Return the Kullback-Leibler divergence
        return min(MAX_ENTROPY_ALLOWED, entropy(p, q)), p,q, lin
    except Exception as e:
        print('Error in KL divergence: ', e)
        return 0.0, None,None, None

def measure_stopping_creteria(pos, neg, minInp):
    frac = pos * 1.0 / (pos + neg)
    print(frac, 2.5 * math.sqrt(frac * (1 - frac) * 1.0 / (pos + neg)), pos+neg, minInp)
    #if 2.5 * math.sqrt(frac * (1 - frac) * 1.0 / (pos + neg) < 0.05) and pos + neg > minInp:
    # break
def get_correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            #print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                #if colname in dataset.columns:
                    #del dataset[colname] # deleting the column from the dataset
    return col_corr