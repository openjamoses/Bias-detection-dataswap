from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
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

class FairnessMetrics_partioning:
    def __init__(self, column_data, choice_dict, protected_key=0.0, non_protected_key=1.0, fav=1.0, non_fav=0.0):
        self.column_data = column_data
        self.choice_dict = choice_dict
        self.protected_key = protected_key
        self.non_protected_key = non_protected_key
        self.fav = fav
        self.non_fav = non_fav
    def extract_subgroup_stats(self, y_original, y_predicted):
        TP_1 = 0
        FP_1 = 0
        TN_1 = 0
        FN_1 = 0

        FN_0 = 0
        FP_0 = 0
        TP_0 = 0
        TN_0 = 0
        for i in range(len(self.column_data)):
            if self.column_data[i] in self.choice_dict[self.protected_key]:
                if y_predicted[i] == self.fav and y_predicted[i] == y_original[i]:
                    TP_0 += 1
                elif y_predicted[i] == self.fav and y_predicted[i] != y_original[i]:
                    FP_0 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] == y_original[i]:
                    TN_0 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] != y_original[i]:
                    FN_0 += 1
            elif self.column_data[i] in self.choice_dict[self.non_protected_key]:
                if y_predicted[i] == self.fav and y_predicted[i] == y_original[i]:
                    TP_1 += 1
                elif y_predicted[i] == self.fav and y_predicted[i] != y_original[i]:
                    FP_1 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] == y_original[i]:
                    TN_1 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] != y_original[i]:
                    FN_1 += 1
        return TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0
    def SPD(self, y_original, y_predicted):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        P1, P0 = 0, 0
        if (TP_1 + TN_1 + FN_1 + FP_1) > 0:
            P1 = (TP_1 + FP_1) / (TP_1 + TN_1 + FN_1 + FP_1)
        if (TP_0 + TN_0 + FN_0 + FP_0) > 0:
            P0 = (TP_0 + FP_0) / (TP_0 + TN_0 + FN_0 + FP_0)
        SPD = (P0 - P1)
        return round(abs(SPD), 3)
    def DIR(self, y_original, y_predicted):
        # P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
        # P_female =  (TP_female + FP_female)/(TP_female + TN_female + FN_female +  FP_female)
        pos_0 = 0
        neg_0 = 0

        pos_1 = 0
        neg_1 = 0
        for i in range(len(self.column_data)):
            if self.column_data[i] in self.choice_dict[self.protected_key]:
                if y_predicted[i] == self.fav:
                    pos_0 += 1
                elif y_predicted[i] == self.non_fav:
                    neg_0 += 1
            if self.column_data[i] in self.choice_dict[self.non_protected_key]:
                if y_predicted[i] == self.fav:
                    pos_1 += 1
                elif y_predicted[i] == self.non_fav:
                    neg_1 += 1
        P0 = 0
        P1 = 0
        if pos_0+neg_0 > 0:
            P0 = pos_0 / (pos_0 + neg_0)
        if (pos_1 + neg_1) > 0:
            P1 = pos_1 / (pos_1 + neg_1)
        #TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        DI = 0
        if P1 > 0:
            DI = (P0 / P1)
        return round(DI, 3)

    def TPR_difference(self, y_original, y_predicted):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        # TPR_male = TP_male/(TP_male+FN_male)
        # TPR_female = TP_female/(TP_female+FN_female)
        TPR_0, TPR_1 = 0, 0
        if (TP_0+FN_0) > 0:
            TPR_0 = TP_0/(TP_0+FN_0)
        if (TP_1+FN_1) > 0:
            TPR_1 = TP_1/(TP_1+FN_1)
        diff = (TPR_1 - TPR_0)
        return round(diff, 3)
    def FPR_difference(self, y_original, y_predicted):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        #FPR_male = FP_male / (FP_male + TN_male)
        # FPR_female = FP_female/(FP_female+TN_female)
        FPR_0, FPR_1 = 0, 0
        if (FP_0+TN_0) > 0:
            FPR_0 = FP_0/(FP_0+TN_0)
        if (FP_1+TN_1) > 0:
            FPR_1 = FP_1/(FP_1+TN_1)
        diff = (FPR_0 - FPR_1)
        return round(diff, 3)
    def AOD(self, y_original, y_predicted):
        FPR_diff = self.FPR_difference(y_original, y_predicted)
        TPR_diff = self.TPR_difference(y_original, y_predicted)
        average_odds_difference = (FPR_diff + TPR_diff) / 2
        return round(average_odds_difference, 3)


    def fairness_metrics(self, y_original, y_predicted):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        print(TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0, np.unique(self.column_data), self.protected_key, self.non_protected_key, self.choice_dict)
        FPR_0, FPR_1 = 0, 0
        if (FP_0 + TN_0) > 0:
            FPR_0 = FP_0 / (FP_0 + TN_0)
        if (FP_1 + TN_1) > 0:
            FPR_1 = FP_1 / (FP_1 + TN_1)
        diff = (FPR_0 - FPR_1)
        FPR_difference = round(diff, 3)

        TPR_0, TPR_1 = 0, 0
        if (TP_0 + FN_0) > 0:
            TPR_0 = TP_0 / (TP_0 + FN_0)
        if (TP_1 + FN_1) > 0:
            TPR_1 = TP_1 / (TP_1 + FN_1)
        diff = (TPR_1 - TPR_0)
        TPR_difference = round(diff, 3)

        AOD = (FPR_difference + TPR_difference) / 2
        #print(y_original, y_predicted)
        P1, P0 = 0, 0
        if (TP_1 + TN_1 + FN_1 + FP_1) > 0:
            P1 = (TP_1 + FP_1) / (TP_1 + TN_1 + FN_1 + FP_1)
        if (TP_0 + TN_0 + FN_0 + FP_0) > 0:
            P0 = (TP_0 + FP_0) / (TP_0 + TN_0 + FN_0 + FP_0)
        SPD = (P0 - P1)
        SPD = round(abs(SPD), 3)

        DIR = self.DIR(y_original, y_predicted)

        cm = confusion_matrix(y_original, y_predicted)
        TN, FP, FN, TP = cm.ravel()

        TPR = calculate_TPR(TP, FP, TN, FN)
        FPR = calculate_FPR(TP, FP, TN, FN)
        SP = calculate_SP(TP, FP, TN, FN)

        Precision = calculate_Precision(TP, FP, TN, FN)
        Recal = calculate_Recall(TP, FP, TN, FN)
        F1 = calculate_F1(TP, FP, TN, FN)
        ACC = calculate_Accuracy(TP, FP, TN, FN)

        return TPR_difference, FPR_difference, SPD, DIR, AOD, TPR, FPR, SP, Precision, Recal, F1, ACC


class FairnessMetrics_binary:
    def __init__(self, column_data, protected_key=0.0, non_protected_key=1.0, fav=1.0, non_fav=0.0):
        self.column_data = column_data
        self.protected_key = protected_key
        self.non_protected_key = non_protected_key
        self.fav = fav
        self.non_fav = non_fav
    def extract_subgroup_stats(self, y_original, y_predicted):
        TP_1 = 0
        FP_1 = 0
        TN_1 = 0
        FN_1 = 0

        FN_0 = 0
        FP_0 = 0
        TP_0 = 0
        TN_0 = 0

        print('self.column_data: ', type(self.column_data))
        for i in range(len(self.column_data)):
            if self.column_data[i] == self.protected_key:
                if y_predicted[i] == self.fav and y_predicted[i] == y_original[i]:
                    TP_0 += 1
                elif y_predicted[i] == self.fav and y_predicted[i] != y_original[i]:
                    FP_0 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] == y_original[i]:
                    TN_0 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] != y_original[i]:
                    FN_0 += 1
            elif self.column_data[i] == self.non_protected_key:
                if y_predicted[i] == self.fav and y_predicted[i] == y_original[i]:
                    TP_1 += 1
                elif y_predicted[i] == self.fav and y_predicted[i] != y_original[i]:
                    FP_1 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] == y_original[i]:
                    TN_1 += 1
                elif y_predicted[i] == self.non_fav and y_predicted[i] != y_original[i]:
                    FN_1 += 1
        return TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0
    def SPD(self, y_original, y_predicted):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        P1, P0 = 0, 0
        if (TP_0 + TN_0 + FN_0 + FP_0) > 0:
            P0 = (TP_0 + FP_0) / (TP_0 + TN_0 + FN_0 + FP_0)
        if (TP_1 + TN_1 + FN_1 + FP_1) > 0:
            P1 = (TP_1 + FP_1) / (TP_1 + TN_1 + FN_1 + FP_1)
        SPD = (P0 - P1)
        return round(abs(SPD), 3)
    def DIR(self, y_original, y_predicted):
        # P_male = (TP_male + FP_male)/(TP_male + TN_male + FN_male + FP_male)
        # P_female =  (TP_female + FP_female)/(TP_female + TN_female + FN_female +  FP_female)
        pos_0 = 0
        neg_0 = 0

        pos_1 = 0
        neg_1 = 0
        for i in range(len(self.column_data)):
            if self.column_data[i] == self.protected_key:
                if y_predicted[i] == self.fav:
                    pos_0 += 1
                elif y_predicted[i] == self.non_fav:
                    neg_0 += 1
            if self.column_data[i] == self.non_protected_key:
                if y_predicted[i] == self.fav:
                    pos_1 += 1
                elif y_predicted[i] == self.non_fav:
                    neg_1 += 1
        P0 = 0
        P1 = 0
        if pos_0+neg_0 > 0:
            P0 = pos_0 / (pos_0 + neg_0)
        if (pos_1 + neg_1) > 0:
            P1 = pos_1 / (pos_1 + neg_1)
        #TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        DI = 0
        if P1 > 0:
            DI = (P0 / P1)
        return round(DI, 3)

    def TPR_difference(self, y_original, y_predicted):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        # TPR_male = TP_male/(TP_male+FN_male)
        # TPR_female = TP_female/(TP_female+FN_female)
        TPR_0, TPR_1 = 0, 0
        if (TP_0+FN_0) > 0:
            TPR_0 = TP_0/(TP_0+FN_0)
        if (TP_1+FN_1) > 0:
            TPR_1 = TP_1/(TP_1+FN_1)
        diff = (TPR_1 - TPR_0)
        return round(diff, 3)
    def FPR_difference(self, y_original, y_predicted):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        #FPR_male = FP_male / (FP_male + TN_male)
        # FPR_female = FP_female/(FP_female+TN_female)
        FPR_0, FPR_1 = 0, 0
        if (FP_0+TN_0) > 0:
            FPR_0 = FP_0/(FP_0+TN_0)
        if (FP_1+TN_1) > 0:
            FPR_1 = FP_1/(FP_1+TN_1)
        diff = (FPR_0 - FPR_1)
        return round(diff, 3)
    def AOD(self, y_original, y_predicted):
        FPR_diff = self.FPR_difference(y_original, y_predicted)
        TPR_diff = self.TPR_difference(y_original, y_predicted)
        average_odds_difference = (FPR_diff + TPR_diff) / 2
        return round(average_odds_difference, 3)


    def fairness_metrics(self, y_original, y_predicted, labels=[0., 1.0]):
        TP_1, FP_1, TN_1, FN_1, FN_0, FP_0, TP_0, TN_0 = self.extract_subgroup_stats(y_original, y_predicted)
        FPR_0, FPR_1 = 0, 0
        if (FP_0 + TN_0) > 0:
            FPR_0 = FP_0 / (FP_0 + TN_0)
        if (FP_1 + TN_1) > 0:
            FPR_1 = FP_1 / (FP_1 + TN_1)
        diff = (FPR_0 - FPR_1)
        FPR_difference = round(diff, 3)

        TPR_0, TPR_1 = 0, 0
        if (TP_0 + FN_0) > 0:
            TPR_0 = TP_0 / (TP_0 + FN_0)
        if (TP_1 + FN_1) > 0:
            TPR_1 = TP_1 / (TP_1 + FN_1)
        diff = (TPR_1 - TPR_0)
        TPR_difference = round(diff, 3)

        AOD = (FPR_difference + TPR_difference)/2

        P1, P0 = 0, 0
        if (TP_1 + TN_1 + FN_1 + FP_1) > 0:
            P1 = (TP_1 + FP_1) / (TP_1 + TN_1 + FN_1 + FP_1)
        if (TP_0 + TN_0 + FN_0 + FP_0) > 0:
            P0 = (TP_0 + FP_0) / (TP_0 + TN_0 + FN_0 + FP_0)
        SPD = (P0 - P1)
        SPD = round(abs(SPD), 3)

        DIR = self.DIR(y_original, y_predicted)

        cm = confusion_matrix(y_original, y_predicted)
        #print(cm.ravel(), y_original.shape, y_predicted.shape, np.unique(y_predicted), np.unique(y_original))
        #print(cm.ravel())
        try:
            TN, FP, FN, TP = cm.ravel()
        except Exception as e:
            TN, FP, FN, TP = 0,0,0,0
            for i in range(len(y_original)):
                if y_predicted[i] == 1 and y_original[i] == 1:
                    TP += 1
                if y_predicted[i] == 1 and y_original[i] == 0:
                    FP += 1
                if y_predicted[i] == 0 and y_original[i] == 0:
                    TN += 1
                if y_predicted[i] == 0 and y_original[i] == 1:
                    FN += 1
            print(' Exception in cross validation: ', e)

        TPR = calculate_TPR(TP, FP, TN, FN)
        FPR = calculate_FPR(TP, FP, TN, FN)
        SP = calculate_SP(TP, FP, TN, FN)

        Precision = calculate_Precision(TP, FP, TN, FN)
        Recal = calculate_Recall(TP, FP, TN, FN)
        F1 = calculate_F1(TP, FP, TN, FN)
        ACC = calculate_Accuracy(TP, FP, TN, FN)

        return TPR_difference, FPR_difference, SPD, DIR, AOD, TPR, FPR, SP, Precision, Recal, F1, ACC