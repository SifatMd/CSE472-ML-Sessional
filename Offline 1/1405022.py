import math
import random


class Node:
    def __init__(self, samples=None, parent_samples=None, attribute_index=None,
                 curr_attribute=None, possible_values=None, plurality=None, depth=None, child_nodes=None, parent_val=None, end=None):
        self.samples = samples
        self.parent_samples = parent_samples
        self.attribute_index = attribute_index
        self.curr_attribute = curr_attribute
        self.possible_values = possible_values
        self.plurality = plurality
        self.depth = depth
        self.child_nodes = child_nodes
        self.parent_val = parent_val
        self.end = end



class DecisionTree:
    def __init__(self, depth):
        self.root = Node()
        self.depth = depth

    @staticmethod
    def calculate_entropy(qval):
        if qval == 0 or qval == 1: return 0
        sums = -(qval * math.log2(qval))
        sums = sums - ((1 - qval) * (math.log2(1 - qval)))
        return sums


    def importance(self, samples, attribute_index):
        positives = [row[-1] for row in samples].count(1)
        x = float(positives / len(samples))
        info_gain = [0,0]
        info_gain[0] = self.calculate_entropy(x)
        max_gain = 0
        for i in attribute_index:
            column = [col[i] for col in samples]
            possible_values = list(set(column))
            #print('possible: ', possible_values)
            sum_entropy = 0
            for ii in possible_values:
                positives = 0
                negatives = 0

                for j in samples:
                    if j[i] == ii:
                        if j[-1]==1.0:
                            positives = positives + 1
                        else:
                            negatives = negatives + 1

                #print(positives, ' ', negatives)
                temp1 = float((positives + negatives) / len(samples))
                temp2 = float(positives/(positives+negatives))
                temp3 = temp1*self.calculate_entropy(temp2)
                sum_entropy += temp3

            #print(info_gain[0], '  ', sum_entropy)
            if (info_gain[0]-sum_entropy) > max_gain:
                max_gain = info_gain[0] - sum_entropy
                info_gain[1] = i

            #print(info_gain)
        #print('index: ', info_gain[1])
        if max_gain == 0:
            ind = random.randrange(len(attribute_index))
            info_gain[1] = attribute_index[ind]

        #print('ww: ', info_gain[1])
        return info_gain[1] #return the attribute index which gives most info gain



    @staticmethod
    def Plurality(samples):
        positives = [row[-1] for row in samples].count(1.0)
        negatives = len(samples) - positives
        if positives > negatives:
            return 1
        else:
            return 0




    def Decision_Tree_Learning(self, samples, attribute_index, parent_samples, curr_depth):
        if len(samples) == 0:
            N = self.Plurality(parent_samples)
            if N==1:
                return Node(plurality=1, end=1)
            else:
                return Node(plurality=0, end=1)

        elif [row[-1] for row in samples].count(1.0)==len(samples):
            return Node(plurality=1, end=1)
        elif [row[-1] for row in samples].count(1.0)==0:
            return Node(plurality=0, end=1)

        elif len(attribute_index)==0 or curr_depth == self.depth:
            N = self.Plurality(samples)
            if N==1:
                return Node(plurality=1, end=1)
            else:
                return Node(plurality=0, end=1)

        else:
            A = self.importance(samples, attribute_index)
            #print(attribute_index, '  --  ', A)
            attribute_index.remove(A)
            N = self.Plurality(samples)

            #print(A)

            column = [row[A] for row in samples]
            possible_values = list(set(column))

            #print(possible_values)

            child_nodes = []
            new_root = Node(samples, parent_samples, attribute_index, A, possible_values , N, curr_depth + 1, child_nodes, end=0)
            copy_samples = samples.copy()
            for i in possible_values:
                sub_samples = []
                ll = len(copy_samples)
                j=0
                for _ in range(ll):
                    if copy_samples[j][A] == i:
                        sub_samples.append(copy_samples[j])
                        copy_samples.pop(j)
                    else:
                        j = j + 1

                #print('samples: ', samples)
                #print('copy: ', copy_samples)

                #print(A, '  ', sub_samples)
                copy_attribute_index = attribute_index.copy()
                child = self.Decision_Tree_Learning(sub_samples, copy_attribute_index, samples, curr_depth + 1)
                child.parent_val = i
                new_root.child_nodes.append(child)

                #print('child parent val: ', child.parent_val)



            return new_root







    def Decision_Tree(self, samples, attribute_index):
        #self.samples = samples
        #self.attribute_index = attribute_index
        #self.depth = depth

        parent_samples = []

        self.root = self.Decision_Tree_Learning(samples, attribute_index, parent_samples, 0)

        return





    def predict(self, test_sample):
        curr_node = self.root

        while True:
            if curr_node.end == 1:
                return curr_node.plurality

            curr_attribute = curr_node.curr_attribute
            #print('attr: ', curr_attribute)
            curr_val = test_sample[curr_attribute]

            flag = 0
            for child in curr_node.child_nodes:
                #print('YO: ', child.parent_val, '  ', curr_val)
                if child.parent_val == curr_val:
                    flag = 1
                    curr_node = child
                    break

            if flag == 0:
                return curr_node.plurality

            #print('end: ', curr_node.end)






    def printTree(self):
        curr_node = self.root
        self.dfs(curr_node)

    def dfs(self, curr_node):

        print(curr_node.end, '  ', curr_node.curr_attribute)
        if curr_node.end==1:
            return

        for i in curr_node.child_nodes:
            self.dfs(i)

        return


import DecisionTreeFirst
import random
import math


class Adaboost:

    def __init__(self, K):
        self.K = K

    def AdaboostImplementation(self, train_data):
        W = []
        h = []
        Z = []

        indices = []
        for i in range(len(train_data[0]) - 1):
            indices.append(i)

        weight = float(1/len(train_data))
        for i in range(len(train_data)):
            W.append(weight)

        for k in range(self.K):
            #Resample
            #print("len train data: ", len(train_data), ' ' , len(W))
            data = random.choices(train_data, weights=W, k=len(train_data))

            #weak learner
            DT = DecisionTreeFirst.DecisionTree(depth=1)
            DT.Decision_Tree(data, indices)
            h.append(DT)

            #print('after DT')
            error = 0

            #calculate error
            lenh = len(h)
            for i in range(len(train_data)):
                yi = h[lenh-1].predict(train_data[i])
                if train_data[i][-1] != yi:
                    error = error + W[i]


            #print('new tree')

            if error > 0.5:
                h.pop()
                k = k - 1
                continue

            #print('error: ', error)
            #update weight
            lenh = len(h)

            if error > 0:
                for i in range(len(train_data)):
                    yi = h[lenh-1].predict(train_data[i])
                    if train_data[i][-1] == yi:
                        W[i] = float(W[i]*((error)/(1-error)))



            #Normalize W
            sums = 0
            for i in W:
                sums += i

            if sums > 0:
                for i in range(len(W)):
                    W[i] = float(W[i]/sums)



            #Calculate weight of this tree
            if error > 0.0:
                err = float((1-error)/error)
                Z.append(math.log2(err))
            elif error==0.0:
                Z.append(10000)
            #print(Z)

        return h, Z











import csv
from sklearn.preprocessing import LabelEncoder
import math
import DecisionTreeFirst
import AdaboostFirst
import random



def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def calculate_average(aray):
    sums = 0
    cnt = 0
    for i in aray:
        if isfloat(i) or isint(i):
            sums += float(i)
            cnt = cnt + 1

    return (float)(sums) / cnt





def test_train_split(dataset, ratio):
    train_len = int(len(dataset)*ratio)
    test_data = dataset.copy()
    train_data = []
    while len(train_data) < train_len:
        index = random.randrange(len(test_data))
        train_data.append(test_data.pop(index))
    return train_data, test_data






def apply_LabelEncoder(dataset, columns_string):
    le = LabelEncoder()
    columns_data = []
    for i in columns_string:
        temp = []
        temp = [item[i] for item in dataset]
        columns_data.append(temp)

    for row in range(len(columns_data)):
        columns_data[row] = le.fit_transform(columns_data[row])

    for i in range(len(columns_data)):
        for j in range(len(columns_data[i])):
            dataset[j][columns_string[i]] = columns_data[i][j]

    # print(dataset)

    return dataset


def dealWithMissingData(dataset):
    columns_string = []
    columns_int = []
    columns_float = []
    common_values_string = []
    common_values_int = []
    common_values_float = []

    attributes = len(dataset[0]) - 1
    common_values = [0] * attributes

    # delete those samples who have missing class values
    for i in range(len(dataset)):
        if dataset[i][attributes] == " " or dataset[i][attributes]=="?":
            dataset.pop(i)

    for i in range(len(dataset[0])):
        if isint(dataset[0][i]):
            columns_int.append(i)
        elif isfloat(dataset[0][i]):
            columns_float.append(i)
        else:
            columns_string.append(i)

    # print(columns_int, columns_float, columns_string)
    # find most common strings
    for i in columns_string:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_string.append(max(temp))

    # find mean values
    # for int
    for i in columns_int:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_int.append((int)(calculate_average(temp)))

    # for float
    for i in columns_float:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_float.append(calculate_average(temp))

    # all common values
    for i in range(attributes):
        if i in columns_string:
            common_values[i] = common_values_string.pop(0)
        elif i in columns_int:
            common_values[i] = common_values_int.pop(0)
        elif i in columns_float:
            common_values[i] = common_values_float.pop(0)

    # print(common_values)
    # assign these values to right places
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] == " " or dataset[i][j] == "?" or dataset[i][j]==" ?" or dataset[i][j]=="? ":
                dataset[i][j] = common_values[j]

    return dataset


def calculate_entropy(q):
    if q == 0 or q == 1: return 0
    sums = -(q * math.log2(q))
    sums = sums - ((1 - q) * (math.log2(1 - q)))
    return sums


def split_into_child(j, dataset):
    left = []
    right = []

    left = dataset[:j]
    right = dataset[j:]

    y0 = 0
    y1 = 0

    x0 = [row[-1] for row in left].count(1.0)
    x1 = [row[-1] for row in right].count(1.0)

    if len(left) != 0: y0 = (float)(x0) / len(left)
    if len(right) != 0: y1 = (float)(x1) / len(right)

    B0 = calculate_entropy(y0) * (len(left) / len(dataset))
    B1 = calculate_entropy(y1) * (len(right) / len(dataset))

    return B0 + B1


def sort_on_column(dataset, column):
    dataset = sorted(dataset, key=lambda x: x[column])
    return dataset


def binarize(dataset, cont_columns):
    info_gain = 0
    copy_dataset = dataset.copy()
    for i in cont_columns:
        x = [row[-1] for row in dataset].count(1)
        y = (float)(x) / len(dataset)
        B = calculate_entropy(y)

        copy_dataset = sort_on_column(copy_dataset, i)
        max_score = [0, 0, 0, 0]

        # take mid value
        for j in range(len(copy_dataset)):
            sums = split_into_child(j, copy_dataset)
            info_gain = B - sums
            # print(info_gain)

            if info_gain > max_score[0]:
                max_score[0] = info_gain
                max_score[1] = i
                max_score[2] = j
                max_score[3] = copy_dataset[j][i]

        # print(max_score[2])
        '''
        for k in range(len(dataset)):
            if k <= max_score[2]:
                dataset[k][i] = 0.0
            else:
                dataset[k][i] = 1.0
        '''
        for k in range(len(dataset)):
            if dataset[k][i] <= max_score[3]:
                dataset[k][i] = 0.0
            else:
                dataset[k][i] = 1.0

                # calculate info gain

                # take the gain with highest score

                # put 0 and 1 in respective classes

    return dataset


def loadDatasetTelco(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    columns_string = []
    dataset = dataset[1:]

    # removing the customer id column
    for i in range(len(dataset)):
        dataset[i] = dataset[i][1:]

    # have to change this later
    #dataset = dataset[:100]

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j].isnumeric():
                dataset[i][j] = (int)(dataset[i][j])
            elif isfloat(dataset[i][j]):
                dataset[i][j] = float(dataset[i][j])

    dataset = dealWithMissingData(dataset)

    # check which columns are string
    for i in range(len(dataset[1])):
        if isinstance(dataset[1][i], str):
            columns_string.append(i)

    dataset = apply_LabelEncoder(dataset, columns_string)

    i = 0
    try:
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    except ValueError:
        print(i)

    # have to binarize
    cont_columns = [4, 17, 18]
    dataset = binarize(dataset, cont_columns)

    return dataset


def preprocess_Telco():
    filename = 'telco-customer-churn/Telco-Customer-Churn.csv'
    dataset = loadDatasetTelco(filename)

    #print(dataset)
    return dataset




dataset = preprocess_Telco()



Ad = AdaboostFirst.Adaboost(20)

train_set, test_set = test_train_split(dataset, 0.80)



H, Z = Ad.AdaboostImplementation(train_set)

print(Z)



correct = 0
for test in test_set:
    result = 0
    for i in range(len(H)):
        h = H[i]
        z = Z[i]

        val = h.predict(test)

        if val == 0:
            val = -1

        result = result + val*z




    if result > 0.0 and test[-1]==1.0:
        correct = correct + 1
    elif result < 0.0 and test[-1]==0.0:
        correct = correct + 1


print("Accuracy: ", correct/len(test_set))




import csv
from sklearn.preprocessing import LabelEncoder
import math
import DecisionTreeFirst
import AdaboostFirst
import random



def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def calculate_average(aray):
    sums = 0
    cnt = 0
    for i in aray:
        if isfloat(i) or isint(i):
            sums += float(i)
            cnt = cnt + 1

    return (float)(sums) / cnt





def test_train_split(dataset, ratio):
    train_len = int(len(dataset)*ratio)
    test_data = dataset.copy()
    train_data = []
    while len(train_data) < train_len:
        index = random.randrange(len(test_data))
        train_data.append(test_data.pop(index))
    return train_data, test_data







def apply_LabelEncoder(dataset, columns_string):
    le = LabelEncoder()
    columns_data = []

    for i in columns_string:
        temp = []
        temp = [item[i] for item in dataset]
        columns_data.append(temp)

    for row in range(len(columns_data)):
        columns_data[row] = le.fit_transform(columns_data[row])

    xx = len(columns_data)
    # print(columns_data[xx-1][:10])


    for i in range(len(columns_data)):
        for j in range(len(columns_data[i])):
            dataset[j][columns_string[i]] = columns_data[i][j]

    # print(dataset)

    return dataset


def dealWithMissingData(dataset):
    columns_string = []
    columns_int = []
    columns_float = []
    common_values_string = []
    common_values_int = []
    common_values_float = []

    attributes = len(dataset[0]) - 1
    common_values = [0] * attributes

    # delete those samples who have missing class values
    for i in range(len(dataset)):
        if dataset[i][attributes] == " " or dataset[i][attributes] == "?":
            dataset.pop(i)

    for i in range(len(dataset[0])):
        if isint(dataset[0][i]):
            columns_int.append(i)
        elif isfloat(dataset[0][i]):
            columns_float.append(i)
        else:
            columns_string.append(i)

    # print(columns_int, columns_float, columns_string)

    # find most common strings
    for i in columns_string:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_string.append(max(temp))

    # find mean values
    # for int
    for i in columns_int:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_int.append((int)(calculate_average(temp)))

    # for float
    for i in columns_float:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_float.append(calculate_average(temp))

    # all common values
    for i in range(attributes):
        if i in columns_string:
            common_values[i] = common_values_string.pop(0)
        elif i in columns_int:
            common_values[i] = common_values_int.pop(0)
        elif i in columns_float:
            common_values[i] = common_values_float.pop(0)

    # print(common_values)
    # assign these values to right places
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] == " " or dataset[i][j] == "?" or dataset[i][j]==" ?" or dataset[i][j]=="? ":
                dataset[i][j] = common_values[j]

    return dataset


def calculate_entropy(q):
    if q == 0 or q == 1: return 0
    sums = -(q * math.log2(q))
    sums = sums - ((1 - q) * (math.log2(1 - q)))
    return sums


def split_into_child(j, dataset):
    left = []
    right = []

    left = dataset[:j]
    right = dataset[j:]

    y0 = 0
    y1 = 0

    x0 = [row[-1] for row in left].count(1.0)
    x1 = [row[-1] for row in right].count(1.0)

    if len(left) != 0: y0 = (float)(x0) / len(left)
    if len(right) != 0: y1 = (float)(x1) / len(right)

    B0 = calculate_entropy(y0) * (len(left) / len(dataset))
    B1 = calculate_entropy(y1) * (len(right) / len(dataset))

    return B0 + B1


def sort_on_column(dataset, column):
    dataset = sorted(dataset, key=lambda x: x[column])
    return dataset


def binarize(dataset, cont_columns):
    info_gain = 0
    copy_dataset = dataset.copy()
    for i in cont_columns:
        x = [row[-1] for row in dataset].count(1)
        y = (float)(x) / len(dataset)
        B = calculate_entropy(y)

        copy_dataset = sort_on_column(copy_dataset, i)
        max_score = [0, 0, 0, 0]

        # take mid value
        for j in range(len(copy_dataset)):
            sums = split_into_child(j, copy_dataset)
            info_gain = B - sums
            # print(info_gain)

            if info_gain > max_score[0]:
                max_score[0] = info_gain
                max_score[1] = i
                max_score[2] = j
                max_score[3] = copy_dataset[j][i]

        # print(max_score[2])
        for k in range(len(dataset)):
            if dataset[k][i] <= max_score[3]:
                dataset[k][i] = 0.0
            else:
                dataset[k][i] = 1.0


                # calculate info gain

                # take the gain with highest score

                # put 0 and 1 in respective classes

    return dataset


def loadDatasetAdult(filename1, filename2):
    lines = csv.reader(open(filename1, "rt"))
    lines2 = csv.reader(open(filename2, "rt"))

    dataset = list(lines)
    dataset2 = list(lines2)
    dataset2 = dataset2[1:]

    dataset = dataset[:10000]
    dataset2 = dataset2[:2500]

    columns_string = []

    len1 = len(dataset)

    for i in dataset2:
        dataset.append(i)

        # have to change this later
        # if filename=="second/Adult_Test.csv":
        #   dataset=dataset[1:]

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j].isnumeric():
                dataset[i][j] = (int)(dataset[i][j])
            elif isfloat(dataset[i][j]):
                dataset[i][j] = float(dataset[i][j])


    #print('deal with missing data')
    dataset = dealWithMissingData(dataset)

    # check which columns are string
    for i in range(len(dataset[1])):
        if isinstance(dataset[1][i], str):
            columns_string.append(i)

    # print(columns_string)
    dataset = apply_LabelEncoder(dataset, columns_string)

    i = 0
    try:
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    except ValueError:
        print(i)

    # have to binarize
    #print('binarize')
    cont_columns = [0, 2, 4, 10, 11, 12]
    dataset = binarize(dataset, cont_columns)

    print('binarize complete')

    dataset2 = dataset[len1:]
    dataset = dataset[:len1]

    return dataset, dataset2


def preprocess_Adult():
    filename1 = 'second/Adult_Train.csv'
    filename2 = 'second/Adult_Test.csv'
    train_data, test_data = loadDatasetAdult(filename1, filename2)

    # print('\n')
    #print(train_data[3:10])
    #print("-------------------------")
    #print(test_data[3:10])

    return train_data, test_data



train_set, test_set = preprocess_Adult()




DT = DecisionTreeFirst.DecisionTree(depth=30)

indices = []
for i in range(len(train_set[0]) - 1):
    indices.append(i)

DT.Decision_Tree(train_set, indices)


correct = 0

true_positive = 0
condition_positive = 0
true_negative = 0
condition_negative = 0
predicted_cond_pos = 0
false_positive = 0


for i in train_set:
    res = DT.predict(i)

    if res == i[-1]:
        correct += 1

    if res==1 and i[-1]==1:
        true_positive += 1

    if i[-1]==1:
        condition_positive += 1

    if res==0 and i[-1]==0:
        true_negative += 1

    if i[-1]==0:
        condition_negative += 1

    if res == 1:
        predicted_cond_pos += 1

    if res == 1 and i[-1]==0:
        false_positive += 1



recall = float(true_positive/condition_positive)
precision = float(true_positive/predicted_cond_pos)

print('Accuracy: ', float(correct/len(train_set)))

print('Recall: ', float(true_positive/condition_positive))
print('Specificity: ', float(true_negative/condition_negative))
print('Precision: ', float(true_positive/predicted_cond_pos))
print('False Discovery Rate: ', float(false_positive/predicted_cond_pos))

yy = float((1/recall)+(1/precision))
print('F1 score: ', float(2/yy))


















'''

Ad = AdaboostFirst.Adaboost(20)


H, Z = Ad.AdaboostImplementation(train_set)

print(Z)

print('Prediction:')

correct = 0
for test in test_set:
    result = 0
    for i in range(len(H)):
        h = H[i]
        z = Z[i]

        val = h.predict(test)

        if val == 0:
            val = -1

        result = result + val*z



    if result > 0.0 and test[-1]==1.0:
        correct = correct + 1
    elif result < 0.0 and test[-1]==0.0:
        correct = correct + 1


print("Accuracy: ", correct/len(test_set))



'''

import csv
from sklearn.preprocessing import LabelEncoder
import math
import DecisionTreeFirst
import AdaboostFirst
import random


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def calculate_average(aray):
    sums = 0
    cnt = 0
    for i in aray:
        if isfloat(i) or isint(i):
            sums += float(i)
            cnt = cnt + 1

    return (float)(sums) / cnt


def test_train_split(dataset, ratio):
    train_len = int(len(dataset) * ratio)
    train_data = dataset[:train_len]
    test_data = dataset[train_len:]

    return train_data, test_data


def apply_LabelEncoder(dataset, columns_string):
    le = LabelEncoder()
    columns_data = []
    for i in columns_string:
        temp = []
        temp = [item[i] for item in dataset]
        columns_data.append(temp)

    for row in range(len(columns_data)):
        columns_data[row] = le.fit_transform(columns_data[row])

    for i in range(len(columns_data)):
        for j in range(len(columns_data[i])):
            dataset[j][columns_string[i]] = columns_data[i][j]

    # print(dataset)

    return dataset


def dealWithMissingData(dataset):
    columns_string = []
    columns_int = []
    columns_float = []
    common_values_string = []
    common_values_int = []
    common_values_float = []

    attributes = len(dataset[0]) - 1
    common_values = [0] * attributes

    # delete those samples who have missing class values
    for i in range(len(dataset)):
        if dataset[i][attributes] == " " or dataset[i][attributes] == "?":
            dataset.pop(i)

    '''
    for i in range(len(dataset[0])):
        if isint(dataset[0][i]):
            columns_int.append(i)
        elif isfloat(dataset[0][i]):
            columns_float.append(i)
        else:
            columns_string.append(i)
    '''
    columns_int.append(0)
    columns_int.append(30)

    for i in range(1, 30, 1):
        columns_float.append(i)

    # find most common strings
    for i in columns_string:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_string.append(max(temp))

    # find mean values
    # for int
    for i in columns_int:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_int.append((int)(calculate_average(temp)))

    # for float
    for i in columns_float:
        temp = []
        temp = [item[i] for item in dataset]
        common_values_float.append(calculate_average(temp))

    # all common values
    for i in range(attributes):
        if i in columns_string:
            common_values[i] = common_values_string.pop(0)
        elif i in columns_int:
            common_values[i] = common_values_int.pop(0)
        elif i in columns_float:
            common_values[i] = common_values_float.pop(0)

    # print(common_values)
    # assign these values to right places
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] == " " or dataset[i][j] == "?":
                dataset[i][j] = common_values[j]

    return dataset


def calculate_entropy(q):
    if q == 0 or q == 1: return 0

    sums = -(q * math.log2(q))
    sums = sums - ((1 - q) * (math.log2(1 - q)))
    return sums


def split_into_child(j, dataset):
    left = []
    right = []

    left = dataset[:j]
    right = dataset[j:]

    y0 = 0
    y1 = 0

    x0 = [row[-1] for row in left].count(1.0)
    x1 = [row[-1] for row in right].count(1.0)

    if len(left) != 0: y0 = (float)(x0) / len(left)
    if len(right) != 0: y1 = (float)(x1) / len(right)

    B0 = calculate_entropy(y0) * (len(left) / len(dataset))
    B1 = calculate_entropy(y1) * (len(right) / len(dataset))

    return B0 + B1


def sort_on_column(dataset, column):
    dataset = sorted(dataset, key=lambda x: x[column])
    return dataset


def binarize(dataset, cont_columns):
    info_gain = 0
    copy_dataset = dataset.copy()
    for i in cont_columns:
        x = [row[-1] for row in dataset].count(1)
        y = (float)(x) / len(dataset)
        B = calculate_entropy(y)

        copy_dataset = sort_on_column(copy_dataset, i)
        max_score = [0, 0, 0, 0]

        # take mid value
        for j in range(len(copy_dataset)):
            sums = split_into_child(j, copy_dataset)
            info_gain = B - sums
            # print(info_gain)

            if info_gain > max_score[0]:
                max_score[0] = info_gain
                max_score[1] = i
                max_score[2] = j
                max_score[3] = copy_dataset[j][i]

        # print(max_score[2])
        for k in range(len(dataset)):
            if dataset[k][i] <= max_score[3]:
                dataset[k][i] = 0.0
            else:
                dataset[k][i] = 1.0
                # calculate info gain

                # take the gain with highest score

                # put 0 and 1 in respective classes

    return dataset


def loadDatasetCreditFraud(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    columns_string = []
    dataset = dataset[1:]

    num_neg = 0
    num_pos = 0
    pos_index = 0
    only_positives = []
    dataset_copy = []
    '''
    for i in dataset:
        if i[-1] == "0" and num_neg < 5000:
            dataset_copy.append(i)
            num_neg = num_neg + 1
        elif i[-1] == "1":
            dataset_copy.append(i)
    '''
    for i in dataset:
        if i[-1] == "1" and num_pos < 400:
            dataset_copy.append(i)
            num_pos = num_pos + 1

        elif i[-1] == "1" and num_pos >= 400:
            only_positives.append(i)
            num_pos = num_pos + 1

    while num_neg < 10000:
        index = random.randrange(len(dataset))
        if dataset[index][-1] == "0":
            dataset_copy.append(dataset.pop(index))
            num_neg = num_neg + 1

    # print('ll: ', len(only_positives))
    for i in only_positives:
        dataset_copy.append(i)

    dataset = dataset_copy.copy()

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j].isnumeric():
                dataset[i][j] = (int)(dataset[i][j])
            elif isfloat(dataset[i][j]):
                dataset[i][j] = float(dataset[i][j])

    dataset = dealWithMissingData(dataset)

    # check which columns are string
    for i in range(len(dataset[1])):
        if isinstance(dataset[1][i], str):
            columns_string.append(i)

    # print(columns_string)
    # dataset = apply_LabelEncoder(dataset, columns_string)

    i = 0
    try:
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    except ValueError:
        print(i)

    # have to binarize
    cont_columns = []
    for i in range(30):  # 30th column is the class values, so it is not continuous
        cont_columns.append(i)
    dataset = binarize(dataset, cont_columns)

    print('binarization done')

    return dataset


def preprocess_CreditFraud():
    filename = 'creditcardfraud/creditcard.csv'
    dataset = loadDatasetCreditFraud(filename)

    # print(len(dataset))
    return dataset


dataset = preprocess_CreditFraud()

train_set, test_set = test_train_split(dataset, 0.8)

random.shuffle(train_set)
random.shuffle(test_set)

DT = DecisionTreeFirst.DecisionTree(depth=40)

indices = []
for i in range(len(train_set[0]) - 1):
    indices.append(i)

DT.Decision_Tree(train_set, indices)

correct = 0

true_positive = 0
condition_positive = 0
true_negative = 0
condition_negative = 0
predicted_cond_pos = 0
false_positive = 0

for i in train_set:
    res = DT.predict(i)

    if res == i[-1]:
        correct += 1

    if res == 1 and i[-1] == 1:
        true_positive += 1

    if i[-1] == 1:
        condition_positive += 1

    if res == 0 and i[-1] == 0:
        true_negative += 1

    if i[-1] == 0:
        condition_negative += 1

    if res == 1:
        predicted_cond_pos += 1

    if res == 1 and i[-1] == 0:
        false_positive += 1

recall = float(true_positive / condition_positive)
precision = float(true_positive / predicted_cond_pos)

print('Accuracy: ', float(correct / len(train_set)))

print('Recall: ', float(true_positive / condition_positive))
print('Specificity: ', float(true_negative / condition_negative))
print('Precision: ', float(true_positive / predicted_cond_pos))
print('False Discovery Rate: ', float(false_positive / predicted_cond_pos))

yy = float((1 / recall) + (1 / precision))
print('F1 score: ', float(2 / yy))

'''

Ad = AdaboostFirst.Adaboost(20)

H, Z = Ad.AdaboostImplementation(train_set)

print(Z)



print('Prediction ')

correct = 0
positives = 0
for test in test_set:
    result = 0
    for i in range(len(H)):
        h = H[i]
        z = Z[i]

        val = h.predict(test)

        if val == 0:
            val = -1

        result = result + val*z




    if result > 0.0 and test[-1]==1.0:
        correct = correct + 1
        positives = positives + 1
    elif result < 0.0 and test[-1]==0.0:
        correct = correct + 1


print('Positive: ', positives)
print("Accuracy: ", correct/len(test_set))


'''













































































































































