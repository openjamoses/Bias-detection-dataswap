import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, KBinsDiscretizer


class LoadData:
    def __init__(self, path=None, threshold=None):
        self.path = path
        #self.protected_attribute = 'sex'
        self.sensitive_list = []
        self.sensitive_indices = []
        self.target_name = None
        self.target_index = None
        self.threshold = threshold
        self.apply_threshold = False
        if threshold != None:
            self.apply_threshold = True #header=0,
    def read_csv(self, filename):
        return pd.read_csv(self.path+filename).dropna()
    def discretized(self, data):
        kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
        return kbins.fit_transform(data)

    def get_correlation(self, dataset, threshold, custom_colums=None):
        col_corr = set()  # Set of all the names of deleted columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                #print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
                    if colname in dataset.columns and colname != self.target_name:
                        del dataset[colname] # deleting the column from the dataset
        if custom_colums != None:
            for column in custom_colums:
                if column in dataset.columns and column != self.target_name:
                    del dataset[column]  # deleting the column from the dataset
        sensitive_list = []
        sensitive_indices = []
        for sensitive in self.sensitive_list:
            if sensitive in dataset.columns.tolist():
                sensitive_list.append(sensitive)
                sensitive_indices.append(dataset.columns.tolist().index(sensitive))

        self.sensitive_list = []
        self.sensitive_indices = []
        for i in range(len(sensitive_list)):
            self.sensitive_list.append(sensitive_list[i])
            self.sensitive_indices.append(sensitive_indices[i])
        self.target_index = dataset.columns.tolist().index(self.target_name)
        print('Correlated columns removed: ', col_corr)
        return dataset
    def load_adult_data(self, filename):
        df = self.read_csv(filename)
        df = df.drop(
            ['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'],
            axis=1)
        ## Change symbolics to numerics
        df['sex'] = np.where(df['sex'] == ' Male', 1, 0)
        df['race'] = np.where(df['race'] != ' White', 0, 1)
        df['Probability'] = np.where(df['Probability'] == ' <=50K', 0, 1)

        ## Discretize age
        '''df['age'] = np.where(df['age'] >= 70, 70, df['age'])
        df['age'] = np.where((df['age'] >= 60) & (df['age'] < 70), 60,
                                       df['age'])
        df['age'] = np.where((df['age'] >= 50) & (df['age'] < 60), 50,
                                       df['age'])
        df['age'] = np.where((df['age'] >= 40) & (df['age'] < 50), 40,
                                       df['age'])
        df['age'] = np.where((df['age'] >= 30) & (df['age'] < 40), 30,
                                       df['age'])
        df['age'] = np.where((df['age'] >= 20) & (df['age'] < 30), 20,
                                       df['age'])
        df['age'] = np.where((df['age'] >= 10) & (df['age'] < 10), 10, df['age'])
        df['age'] = np.where(df['age'] < 10, 0, df['age'])'''
        #self.protected_attribute = 'sex'

        self.sensitive_list = ['sex', 'race', 'age']
        self.sensitive_indices = [df.columns.tolist().index('sex'),df.columns.tolist().index('race')]
        self.target_name = 'Probability'
        self.target_index = df.columns.tolist().index(self.target_name)

        # Based on class
        #adult_df_one, adult_df_zero = [x for _, x in adult_df.groupby(adult_df['Probability'] == 0)]

        # Based on sex
        #adult_df_one_male, adult_df_one_female = [x for _, x in adult_df_one.groupby(adult_df_one['sex'] == 0)]
        #adult_df_zero_male, adult_df_zero_female = [x for _, x in adult_df_zero.groupby(adult_df_zero['sex'] == 0)]

        # Based on race
        #adult_df_one_white, adult_df_one_nonwhite = [x for _, x in adult_df_one.groupby(adult_df_one['race'] == 0)]
        #adult_df_zero_white, adult_df_zero_nonwhite = [x for _, x in adult_df_zero.groupby(adult_df_zero['race'] == 0)]

        #print(adult_df_one_male.shape, adult_df_one_female.shape, adult_df_zero_male.shape, adult_df_zero_female.shape)
        #print(adult_df_one_white.shape, adult_df_one_nonwhite.shape, adult_df_zero_white.shape,
        #      adult_df_zero_nonwhite.shape)
        if self.apply_threshold:
            #'education-num'
            custom_columns = ['education-num']
            df = self.get_correlation(df, self.threshold, custom_colums=custom_columns)
        return df
    def load_compas_data(self, filename, drop_feature=[]):
        ## Load dataset
        #df = self.read_csv(filename)
        df = pd.read_csv(self.path + filename)
        #df = pd.read_csv('../data/compas-scores-two-years.csv')
        #print('before droped: ', df)

        drop_these = ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age_cat', 'juv_fel_count', 'decile_score',
             'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
             'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
             'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
             'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
             'vr_charge_desc', 'type_of_assessment', 'decile_score', 'score_text', 'screening_date',
             'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody', 'out_custody',
             'start', 'end', 'event']
        if len(drop_feature) > 0:
            drop_these.extend(drop_feature)
        ## Drop categorical features
        ## Removed two duplicate coumns - 'decile_score','priors_count'


        #print('after droped: ', df)
        ## Change symbolics to numerics
        df['sex'] = np.where(df['sex'] == 'Female', 1, 0)
        df['race'] = np.where(df['race'] != 'Caucasian', 0, 1)

        df = df.drop(drop_these, axis=1)
        #df['priors_count'] = np.where(
        #    (df['priors_count'] >= 1) & (df['priors_count'] <= 3), 3, df['priors_count'])
        #df['priors_count'] = np.where(df['priors_count'] > 3, 4, df['priors_count'])
        df['priors_count'] = np.where((df['priors_count'] >= 1) & (df['priors_count'] <= 3), 1, 0)
        #df['age_cat'] = np.where(df['age_cat'] == 'Greater than 45', 45, df['age_cat'])
        #df['age_cat'] = np.where(df['age_cat'] == '25 - 45', 25, df['age_cat'])
        #df['age_cat'] = np.where(df['age_cat'] == 'Less than 25', 0, df['age_cat'])
        df['c_charge_degree'] = np.where(df['c_charge_degree'] == 'F', 1, 0)

        self.protected_attribute = 'race'

        ## Rename class column
        df.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

        # Here did not rec means 0 is the favorable lable
        df['Probability'] = np.where(df['Probability'] == 0, 1, 0)
        self.target_name = 'Probability'
        self.target_index = df.columns.tolist().index(self.target_name)

        # Based on class
        #compas_df_one, compas_df_zero = [x for _, x in compas_df.groupby(compas_df['Probability'] == 0)]

        # Based on sex
        #compas_df_one_female, compas_df_one_male = [x for _, x in compas_df_one.groupby(compas_df_one['sex'] == 0)]
        #compas_df_zero_female, compas_df_zero_male = [x for _, x in compas_df_zero.groupby(compas_df_zero['sex'] == 0)]

        # Based on race
        #compas_df_one_caucasian, compas_df_one_notcaucasian = [x for _, x in
        #                                                       compas_df_one.groupby(compas_df_one['race'] == 0)]
        #compas_df_zero_caucasian, compas_df_zero_notcaucasian = [x for _, x in
         #                                                        compas_df_zero.groupby(compas_df_zero['race'] == 0)]

        #print(compas_df_one_female.shape, compas_df_one_male.shape, compas_df_zero_female.shape,
        #     compas_df_zero_male.shape)
        #print(compas_df_one_caucasian.shape, compas_df_one_notcaucasian.shape, compas_df_zero_caucasian.shape,
        #      compas_df_zero_notcaucasian.shape)
        if 'sex' in df.columns.tolist() and 'race' in df.columns.tolist():
            self.sensitive_list = ['sex', 'race']
            self.sensitive_indices = [df.columns.tolist().index('sex'), df.columns.tolist().index('race')]
        elif 'sex' in df.columns.tolist():
            self.sensitive_list = ['sex']
            self.sensitive_indices = [df.columns.tolist().index('sex')]
        elif 'race' in df.columns.tolist():
            self.sensitive_list = ['race']
            self.sensitive_indices = [df.columns.tolist().index('race')]
        else:
            self.sensitive_list = [df.columns.tolist()[0]]
            self.sensitive_indices = [0]
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        if self.apply_threshold:
            df = self.get_correlation(df, self.threshold)
        #print(df)
        df.nunique()
        for col in df:
            print(col, df[col].unique())
        return df

    def load_german_data2(self, filename):
        df = self.read_csv(filename)
        df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
        df['Risk'] = np.where(df['Risk'] == 'good', 1, 0)
        #df['Housing'] = np.where(df['Housing'] == 'free', 0,  df['Housing'])
        #df['Housing'] = np.where(df['Housing'] == 'rent', 1, df['Housing'])
        #df['Housing'] = np.where(df['Housing'] == 'own', 2, df['Housing'])

        self.sensitive_list = ['Sex', 'Age']
        self.sensitive_indices = [df.columns.tolist().index('Sex'), df.columns.tolist().index('Age')]
        self.target_name = 'Risk'
        self.target_index = df.columns.tolist().index(self.target_name)
        df["Saving"].fillna("none", inplace=True)
        df["Checking"].fillna("none", inplace=True)
        df.isna().any()

        enc = OrdinalEncoder()
        tranform_comns = ['Housing',"Saving", 'Checking', 'Purpose']
        enc.fit(df[tranform_comns])
        df[tranform_comns] = enc.transform(df[tranform_comns])

        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df = df.dropna(how='all')
        if self.apply_threshold:
            df = self.get_correlation(df, self.threshold)
        return df


    def load_german_data(self, filename):
        df = self.read_csv(filename)

        df['Probability'] = np.where(df['Probability'] == 2, 0, 1)
        df['sex'] = np.where(df['sex'] == 'A91', 1, df['sex'])
        df['sex'] = np.where(df['sex'] == 'A92', 0, df['sex'])
        df['sex'] = np.where(df['sex'] == 'A93', 1, df['sex'])
        df['sex'] = np.where(df['sex'] == 'A94', 1, df['sex'])
        df['sex'] = np.where(df['sex'] == 'A95', 0, df['sex'])

        # Based on class
        #german_df_one, german_df_zero = [x for _, x in german_df.groupby(german_df['Probability'] == 0)]

        # Based on sex
        #german_df_one_male, german_df_one_female = [x for _, x in german_df_one.groupby(german_df_one['sex'] == 0)]
        #german_df_zero_male, german_df_zero_female = [x for _, x in german_df_zero.groupby(german_df_zero['sex'] == 0)]

        #print(german_df_one_male.shape, german_df_one_female.shape, german_df_zero_male.shape,
        #      german_df_zero_female.shape)

        self.sensitive_list = ['sex', 'age']
        self.sensitive_indices = [df.columns.tolist().index('sex'), df.columns.tolist().index('age')]
        self.target_name = 'Probability'
        self.target_index = df.columns.tolist().index(self.target_name)
        enc = OrdinalEncoder()
        tranform_comns = ["credit_history", '1', '2', '4', '5', 'savings', 'employment', '8','10','11','12','14','15','16','17','18','19','20']
        enc.fit(df[tranform_comns])
        df[tranform_comns] = enc.transform(df[tranform_comns])

        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        if self.apply_threshold:
            df = self.get_correlation(df, self.threshold)
        return df


        return df
    def load_clevelan_heart_data(self, filename, drop_feature=[]):
        df = self.read_csv(filename) #pd.read_csv('../data/processed.cleveland.data.csv')

        df['Probability'] = np.where(df['Probability'] > 0, 1, 0)
        ## calculate mean of age column
        mean = df.loc[:, "age"].mean()
        df['age'] = np.where(df['age'] >= mean, 0, 1)
        if len(drop_feature) > 0:
            df = df.drop(drop_feature, axis=1)
        # #Based on class
        #heart_df_one, heart_df_zero = [x for _, x in heart_df.groupby(heart_df['Probability'] == 0)]

        # Based on age
        #heart_df_one_old, heart_df_one_young = [x for _, x in heart_df_one.groupby(heart_df_one['age'] == 0)]
        #heart_df_zero_old, heart_df_zero_young = [x for _, x in heart_df_zero.groupby(heart_df_zero['age'] == 0)]

        #print(heart_df_one_old.shape, heart_df_one_young.shape, heart_df_zero_old.shape, heart_df_zero_young.shape)

        self.sensitive_list = ['age']
        if 'age' in df.columns.tolist():
            self.sensitive_indices = [df.columns.tolist().index('age')]
        elif 'sex' in df.columns.tolist():
            self.sensitive_indices = [df.columns.tolist().index('sex')]
        else:
            self.sensitive_list = [df.columns.tolist()[0]]
            self.sensitive_indices = [0]

        self.target_name = 'Probability'
        self.target_index = df.columns.tolist().index(self.target_name)

        if self.apply_threshold:
            df = self.get_correlation(df, self.threshold)
        return df
    def load_bank_data(self, filename):
        ## Load dataset
        #from sklearn import preprocessing
        df = self.read_csv(filename) #pd.read_csv('../data/bank.csv')
        df = df.drop(['contact', 'day','marital', 'month', 'pdays', 'previous', 'poutcome'], axis=1)

        ## Drop categorical features

        df['Probability'] = np.where(df['Probability'] == 'yes', 1, 0)
        df['default'] = np.where(df['default'] == 'no', 1, 0)
        df['housing'] = np.where(df['housing'] == 'yes', 1, 0)
        df['loan'] = np.where(df['loan'] == 'no', 1, 0)

        #df['education'] = np.where(df['education'] == 'primary', 1, df['education'])
        #df['education'] = np.where(df['education'] == 'secondary', 2, df['education'])
        df['education'] = np.where(df['education'] == 'tertiary', 1, 0)
        #df['education'] = np.where(df['education'] == 'unknown', 4, df['education'])





        #mean = df.loc[:, "age"].mean()
        #df['age'] = np.where(df['age'] >= 30, 1, 0)
        # Based on class
        #bank_df_one, bank_df_zero = [x for _, x in bank_df.groupby(bank_df['Probability'] == 0)]

        # Based on age
        #bank_df_one_old, bank_df_one_young = [x for _, x in bank_df_one.groupby(bank_df_one['age'] == 0)]
        #bank_df_zero_old, bank_df_zero_young = [x for _, x in bank_df_zero.groupby(bank_df_zero['age'] == 0)]

        #print(bank_df_one_old.shape, bank_df_one_young.shape, bank_df_zero_old.shape, bank_df_zero_young.shape)

        self.sensitive_list = ['age']
        self.sensitive_indices = [df.columns.tolist().index('age')]
        self.target_name = 'Probability'
        self.target_index = df.columns.tolist().index(self.target_name)

        enc = OrdinalEncoder()
        tranform_comns = ["job"]
        enc.fit(df[tranform_comns])
        df[tranform_comns] = enc.transform(df[tranform_comns])

        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        if self.apply_threshold:
            df = self.get_correlation(df, self.threshold)
        return df
    def load_credit_defaulter(self, filename):
        df = self.read_csv(filename)#pd.read_csv('../data/default_of_credit_card_clients_first_row_removed.csv')
        df = df.drop(['ID'], axis=1)

        df['SEX'] = np.where(df['SEX'] == 2, 0, 1)
        #df['SEX'] = np.where(df['SEX'] == 2, 0, 1)

        # Based on class
        #default_df_one, default_df_zero = [x for _, x in df.groupby(df['Probability'] == 0)]

        # Based on sex
        #default_df_one_male, default_df_one_female = [x for _, x in default_df_one.groupby(default_df_one['sex'] == 0)]
        #default_df_zero_male, default_df_zero_female = [x for _, x in
        #                                                default_df_zero.groupby(default_df_zero['sex'] == 0)]

        #print(default_df_one_male.shape, default_df_one_female.shape, default_df_zero_male.shape,
        #      default_df_zero_female.shape)
        ## Change symbolics to numerics
        self.sensitive_list = ['SEX', 'AGE']
        self.sensitive_indices = [df.columns.tolist().index('SEX'), df.columns.tolist().index('AGE')]
        self.target_name = 'default payment next month'
        self.target_index = df.columns.tolist().index(self.target_name)
        #scaler = MinMaxScaler()
        #df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        #print(df)
        if self.apply_threshold:
            df = self.get_correlation(df, self.threshold)
        return df

    def load_home_credit(self, filename):
        #df = self.read_csv(filename) #pd.read_csv('../data/Home Credit Default Risk.csv')
        df = pd.read_csv(self.path+filename)
        df = df.drop(
            ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG',
             'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG','NONLIVINGAREA_AVG',	'APARTMENTS_MODE',	'BASEMENTAREA_MODE',	'YEARS_BEGINEXPLUATATION_MODE',
             'YEARS_BUILD_MODE',	'COMMONAREA_MODE',	'ELEVATORS_MODE',	'ENTRANCES_MODE',	'FLOORSMAX_MODE',	'FLOORSMIN_MODE',	'LANDAREA_MODE',	'LIVINGAPARTMENTS_MODE',	'LIVINGAREA_MODE',	'NONLIVINGAPARTMENTS_MODE',
             'NONLIVINGAREA_MODE',	'APARTMENTS_MEDI',	'BASEMENTAREA_MEDI',	'YEARS_BEGINEXPLUATATION_MEDI',	'YEARS_BUILD_MEDI',	'COMMONAREA_MEDI',	'ELEVATORS_MEDI',	'ENTRANCES_MEDI',	'FLOORSMAX_MEDI',
             'FLOORSMIN_MEDI',	'LANDAREA_MEDI',	'LIVINGAPARTMENTS_MEDI',	'LIVINGAREA_MEDI',	'NONLIVINGAPARTMENTS_MEDI',	'NONLIVINGAREA_MEDI',	'FONDKAPREMONT_MODE',	'HOUSETYPE_MODE',	'TOTALAREA_MODE',
             'WALLSMATERIAL_MODE',	'EMERGENCYSTATE_MODE'], axis=1)


        df['CODE_GENDER'] = np.where(df['CODE_GENDER'] == 'M', 1, 0)
        self.target_name = 'default payment next month'
        # Based on sex
        #home_df_one_male, home_df_one_female = [x for _, x in home_df_one.groupby(home_df_one['CODE_GENDER'] == 0)]
        #home_df_zero_male, home_df_zero_female = [x for _, x in home_df_zero.groupby(home_df_zero['CODE_GENDER'] == 0)]

        #print(home_df_one_male.shape, home_df_one_female.shape, home_df_zero_male.shape, home_df_zero_female.shape)
        self.sensitive_list = ['CODE_GENDER']
        self.sensitive_indices = [df.columns.tolist().index('CODE_GENDER')]
        if self.apply_threshold:
            df = self.get_correlation(df, self.threshold)
        return df
    def load_student_data(self, filename, drop_feature=[]):
        ## Load dataset
        #from sklearn import preprocessing
        df = self.read_csv(filename) #pd.read_csv('../data/student/Student.csv')
        drop_list = ['school', 'famsize', 'address', 'Dalc', 'failures']
        if len(drop_feature)> 0:
            drop_list.extend(drop_feature)
        df = df.drop(drop_list, axis=1) # 'G1', 'G2', 'Walc', 'Mjob', 'Fjob'
        print(drop_list, df.columns)

        df['sex'] = np.where(df['sex'] == 'M', 1, 0)
        df['Probability'] = np.where(df['Probability'] > 12, 1, 0)
        # Based on sex
        #student_df_one_male, student_df_one_female = [x for _, x in student_df_one.groupby(student_df_one['sex'] == 0)]
        #student_df_zero_male, student_df_zero_female = [x for _, x in
        #                                                student_df_zero.groupby(student_df_zero['sex'] == 0)]

        #print(student_df_one_male.shape, student_df_one_female.shape, student_df_zero_male.shape,
        #      student_df_zero_female.shape)
        self.sensitive_list = ['sex']
        self.sensitive_indices = [df.columns.tolist().index('sex')]
        self.target_name = 'Probability'
        self.target_index = df.columns.tolist().index(self.target_name)

        enc = OrdinalEncoder()
        tranform_comns_ = ["Pstatus", 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'Mjob', 'Fjob']
        tranform_comns = []
        for col in tranform_comns_:
            if not col in drop_list:
                tranform_comns.append(col)
        enc.fit(df[tranform_comns])
        df[tranform_comns] = enc.transform(df[tranform_comns])

        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        if self.apply_threshold:
            custom_columns = ['G1']
            df = self.get_correlation(df, self.threshold, custom_colums=custom_columns)
        return df

    def load_meps15(self,filename):
        MEPS15 = self.read_csv(filename) #pd.read_csv('../data/MEPS/h181.csv')

        # ## Drop NULL values
        MEPS15 = MEPS15.dropna()

        MEPS15 = MEPS15.rename(
            columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                     'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                     'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                     'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                     'POVCAT15': 'POVCAT', 'INSCOV15': 'INSCOV'})

        MEPS15 = MEPS15[MEPS15['PANEL'] == 20]
        MEPS15 = MEPS15[MEPS15['REGION'] >= 0]  # remove values -1
        MEPS15 = MEPS15[MEPS15['AGE'] >= 0]  # remove values -1
        MEPS15 = MEPS15[MEPS15['MARRY'] >= 0]  # remove values -1, -7, -8, -9
        MEPS15 = MEPS15[MEPS15['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9
        MEPS15 = MEPS15[
            (MEPS15[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                     'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                     'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                     'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                     'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(1)]

        # ## Change symbolics to numerics
        MEPS15['RACEV2X'] = np.where((MEPS15['HISPANX'] == 2) & (MEPS15['RACEV2X'] == 1), 1, MEPS15['RACEV2X'])
        MEPS15['RACEV2X'] = np.where(MEPS15['RACEV2X'] != 1, 0, MEPS15['RACEV2X'])
        MEPS15 = MEPS15.rename(columns={"RACEV2X": "RACE"})

        # MEPS15['UTILIZATION'] = np.where(MEPS15['UTILIZATION'] >= 10, 1, 0)

        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        MEPS15['TOTEXP15'] = MEPS15.apply(lambda row: utilization(row), axis=1)
        lessE = MEPS15['TOTEXP15'] < 10.0
        MEPS15.loc[lessE, 'TOTEXP15'] = 0.0
        moreE = MEPS15['TOTEXP15'] >= 10.0
        MEPS15.loc[moreE, 'TOTEXP15'] = 1.0

        MEPS15 = MEPS15.rename(columns={'TOTEXP15': 'UTILIZATION'})

        MEPS15 = MEPS15[['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                         'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                         'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                         'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                         'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                         'PCS42', 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION', 'PERWT15F']]

        MEPS15 = MEPS15.rename(columns={"UTILIZATION": "Probability", "RACE": "race"})

        # Based on class
        #MEPS15_one, MEPS15_zero = [x for _, x in MEPS15.groupby(MEPS15['Probability'] == 0)]

        # Based on race
        #MEPS15_one_white, MEPS15_one_nonwhite = [x for _, x in MEPS15_one.groupby(MEPS15_one['race'] == 0)]
        #MEPS15_zero_white, MEPS15_zero_nonwhite = [x for _, x in MEPS15_zero.groupby(MEPS15_zero['race'] == 0)]

        #print(MEPS15_one_white.shape, MEPS15_one_nonwhite.shape, MEPS15_zero_white.shape, MEPS15_zero_nonwhite.shape)
        return MEPS15

    def load_load_mep16(self, filename):
        MEPS16 = self.read_csv(filename) #pd.read_csv('../data/MEPS/h192.csv')

        # ## Drop NULL values
        MEPS16 = MEPS16.dropna()

        MEPS16 = MEPS16.rename(
            columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                     'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                     'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                     'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                     'POVCAT16': 'POVCAT', 'INSCOV16': 'INSCOV'})

        MEPS16 = MEPS16[MEPS16['PANEL'] == 21]
        MEPS16 = MEPS16[MEPS16['REGION'] >= 0]  # remove values -1
        MEPS16 = MEPS16[MEPS16['AGE'] >= 0]  # remove values -1
        MEPS16 = MEPS16[MEPS16['MARRY'] >= 0]  # remove values -1, -7, -8, -9
        MEPS16 = MEPS16[MEPS16['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9
        MEPS16 = MEPS16[
            (MEPS16[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                     'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                     'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                     'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                     'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(1)]

        # ## Change symbolics to numerics
        MEPS16['RACEV2X'] = np.where((MEPS16['HISPANX'] == 2) & (MEPS16['RACEV2X'] == 1), 1, MEPS16['RACEV2X'])
        MEPS16['RACEV2X'] = np.where(MEPS16['RACEV2X'] != 1, 0, MEPS16['RACEV2X'])
        MEPS16 = MEPS16.rename(columns={"RACEV2X": "RACE"})

        # MEPS16['UTILIZATION'] = np.where(MEPS16['UTILIZATION'] >= 10, 1, 0)

        def utilization(row):
            return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']

        MEPS16['TOTEXP16'] = MEPS16.apply(lambda row: utilization(row), axis=1)
        lessE = MEPS16['TOTEXP16'] < 10.0
        MEPS16.loc[lessE, 'TOTEXP16'] = 0.0
        moreE = MEPS16['TOTEXP16'] >= 10.0
        MEPS16.loc[moreE, 'TOTEXP16'] = 1.0

        MEPS16 = MEPS16.rename(columns={'TOTEXP16': 'UTILIZATION'})

        MEPS16 = MEPS16[['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                         'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                         'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                         'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                         'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                         'PCS42', 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION', 'PERWT16F']]

        MEPS16 = MEPS16.rename(columns={"UTILIZATION": "Probability", "RACE": "race"})
        protected_attribute = 'race'

        # Based on class
        #MEPS16_one, MEPS16_zero = [x for _, x in MEPS16.groupby(MEPS16['Probability'] == 0)]

        # Based on race
        #MEPS16_one_white, MEPS16_one_nonwhite = [x for _, x in MEPS16_one.groupby(MEPS16_one['race'] == 0)]
        #MEPS16_zero_white, MEPS16_zero_nonwhite = [x for _, x in MEPS16_zero.groupby(MEPS16_zero['race'] == 0)]

        #print(MEPS16_one_white.shape, MEPS16_one_nonwhite.shape, MEPS16_zero_white.shape, MEPS16_zero_nonwhite.shape)

        return MEPS16
    def _get_sensitive_index(self, df, project_column=None):
        if project_column == None:
            project_column = self.protected_attribute
        return df.columns.tolist().index(project_column)




