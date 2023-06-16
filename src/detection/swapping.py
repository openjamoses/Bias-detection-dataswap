import numpy as np
import random
from src.common.distance_measure import DistanceMeasure_specific


class Swapping:
    def __init__(self, category_indices):
        self.category_indices = category_indices

    def distortion(self, j, m, p, q, dmax=0.2, discrete_indices=[]):
        max_distorted = True
        if j in discrete_indices and m in discrete_indices:
            max_distorted = True
        else:
            distortion = DistanceMeasure_specific._distance_single(p, q, self.category_indices)
            if distortion <= dmax:
                max_distorted = True
            else:
                max_distorted = False
        return max_distorted

    def _alternative(self, posible_choice, current_value):
        choice_1, choice_2 = [], []
        for k, v in posible_choice.items():
            if current_value in v:
                choice_2.append(random.choice(v))
            else:
                choice_1.append(random.choice(v))
        # print(posible_choice, choice_1, choice_2, current_value)
        if len(choice_1) > 0:
            return random.choice(choice_1)
        else:
            return random.choice(choice_2)

    def data_swapping(self, X, column_id, I, choice_dict, dmax=0.2):
        n, m = X.shape
        X_new = np.zeros(shape=(n, m))
        # X_new_temp = np.zeros(n, m)
        for i in range(n):
            X_new[i, :] = X[i, :]
            # X_new_temp[i, :] = X[i, :]
            if i in I:
                X_new[i, column_id] = self._alternative(choice_dict, X[i, column_id])
                if not column_id in self.category_indices:
                    # original_column = get_features_by_index(X, column_id)
                    # swapped_column = get_features_by_index(X_new, column_id)
                    # distance_ = 0
                    distortion_value = DistanceMeasure_specific._distance_single(X_new[i], X[i], self.category_indices)
                    if distortion_value <= dmax:
                        pass
                    else:
                        X_new[i, column_id] = X[i, column_id]
        return X_new

    # TODO: under implementation, please don't use yet!
    def double_swapping(self, X1, X, column_id, column_id2, I, choice_dict, dmax=0.2):
        n, m = X.shape
        X_new = np.zeros(shape=(n, m))
        # X_new_temp = np.zeros(n, m)
        for i in range(n):
            X_new[i, :] = X1[i, :]
            # X_new_temp[i, :] = X[i, :]
            if i in I:
                X_new[i, column_id] = self._alternative(choice_dict, X1[i, column_id])
                max_distorted = self.distortion(column_id, column_id2, X_new[i], X[i], dmax=dmax)
                if max_distorted == False:
                    X_new[i, column_id] = X[i, column_id]
        return X_new
