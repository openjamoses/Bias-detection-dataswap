#todo: example of partial ordering

# Features: race, , -1:['Housing', 'Purpose']
#feature_list, ordering = ['sex','age','health','Pstatus','nursery','Medu', 'Fjob', 'schoolsup', 'absences',  'activities', 'higher', 'traveltime',  'paid', 'guardian',  'Walc', 'freetime', 'famsup',  'romantic', 'studytime', 'goout', 'reason',  'famrel', 'internet'], {0:['sex','age']}
feature_list, ordering = ['Sex', 'Age','Job', 'Saving', 'Checking', 'Credit','Housing', 'Purpose'], {0:['Sex']} #, 'Age', 'Credit'
# TODO: Define constants
VALID_TYPE_DEFINED = 'defined'
VALID_TYPE_FREQUENT = 'frequent'
class Partial_ordering:
    @staticmethod
    def _ordering(ordering, f1_feature, f2_mediator):
        list_taken_starts = []
        list_taken_ = []
        list_taken_first = []
        start_index, random_index = 0, -1
        list_partial_ordering = []
        if start_index in ordering.keys():
            start_orderingL = ordering[start_index]
            list_taken_starts = start_orderingL
            if len(start_orderingL) > 1: # [a, b, c, d, e] ab, bc, cd, de
                for i in range(len(start_orderingL)-1):
                    for j in range(i+1, len(start_orderingL)):
                        list_partial_ordering.append(start_orderingL[i] + '/' + str(start_orderingL[j]))
            for f in start_orderingL:
                list_taken_.append(f)
                list_taken_first.append(f)

        if random_index in ordering.keys():
            random_orderingL = ordering[random_index]
            for i in range(len(random_orderingL) - 1):
                for j in range(i + 1, len(random_orderingL)):
                    list_partial_ordering.append(random_orderingL[i] + '/' + str(random_orderingL[j]))
            for f1 in list_taken_starts:
                for f2 in random_orderingL:
                    list_partial_ordering.append(str(f1)+'/'+str(f2))
            for f2 in random_orderingL:
                list_taken_.append(f2)
        #print(list_partial_ordering)
        if f1_feature in list_taken_ and f2_mediator in list_taken_:
            if str(f1_feature)+'/'+str(f2_mediator) in list_partial_ordering:
                valid, type_ = True, VALID_TYPE_DEFINED
            else:
                valid, type_ = False, VALID_TYPE_DEFINED
        elif f2_mediator in list_taken_first and not f1_feature in list_taken_:
            valid, type_ = False, VALID_TYPE_DEFINED
        elif f1_feature in list_taken_first and not f2_mediator in list_taken_:
            valid, type_ = True, VALID_TYPE_DEFINED
        else:
            valid, type_ = True, VALID_TYPE_FREQUENT
        #print(valid, type_)
        return valid, type_
# if __name__ == '__main__':
#     Partial_ordering._ordering(ordering, 'Sex', 'Credit')
