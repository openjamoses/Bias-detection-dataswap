from __future__ import print_function, division
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors as NN


def get_ngbr(df, knn):
    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]
    ngbr = knn.kneighbors(parent_candidate.values.reshape(1, -1), 3, return_distance=False)
    candidate_1 = df.iloc[ngbr[0][0]]
    candidate_2 = df.iloc[ngbr[0][1]]
    candidate_3 = df.iloc[ngbr[0][2]]
    return parent_candidate, candidate_2, candidate_3


def generate_samples(no_of_samples, df, column_list):
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5, algorithm='auto').fit(df)
    try:
        for _ in range(no_of_samples):
            cr = 0.8
            f = 0.8
            parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
            new_candidate = []
            for key, value in parent_candidate.items():
                if isinstance(parent_candidate[key], bool):
                    new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
                elif isinstance(parent_candidate[key], str):
                    new_candidate.append(
                        random.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]]))
                elif isinstance(parent_candidate[key], list):
                    temp_lst = []
                    for i, each in enumerate(parent_candidate[key]):
                        temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                        int(parent_candidate[key][i] +
                                            f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                    new_candidate.append(temp_lst)
                else:
                    new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))
            total_data.append(new_candidate)

        final_df = pd.DataFrame(total_data)
        column_dict = {}
        for column_id in range(len(column_list)):
            #if column_id != target_index:
            column_dict[column_id] = column_list[column_id]
        final_df = final_df.rename(columns=column_dict)
        return final_df
    except Exception as e:
        print(' Exception in data generation: ', e)
    return df