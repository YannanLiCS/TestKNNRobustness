import random
import numpy as np
from collections import Counter
# to calculate neighbors quickly
from sklearn.neighbors import NearestNeighbors


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


    
def getLabelCnt(nabrs_indices, y_train, k):
    targets = [y_train[i] for i in nabrs_indices]
    return Counter(targets)

    
def getFreqLabel(cntr):
    [ori_label, ori_cnt] = cntr.most_common(1)[0]
    if len(cntr.most_common(2)) < 2:
        return ori_label
    for label, cnt in cntr.items():
        if label != ori_label and cnt == ori_cnt:
            ori_label = min(ori_label, label)
    return ori_label
 
 
def getFstSndFreqLabelCnt(cntr, ori_label):
    if len(cntr.most_common(2)) < 2:
        if ori_label > 0:
            return [cntr.most_common(1)[0][1], 0, -1]
        else:
            return [cntr.most_common(1)[0][1], 0, 1]
    scd_cnt = 0
    for label, cnt in cntr.items():
        if label == ori_label:
            fst_cnt = cnt
        else:
            if cnt > scd_cnt:
                scd_cnt = cnt
                scd_label = label
    return fst_cnt, scd_cnt, scd_label
    


import math
def canGetAnotherLabel(counter_k, m_removal):
    ori_label = getFreqLabel(counter_k)
    fst_cnt, scd_cnt, scd_label = getFstSndFreqLabelCnt(counter_k, ori_label)
    return math.ceil((fst_cnt - scd_cnt + (ori_label < scd_label)) / 2) <= m_removal
    
    

def predict_with_label(nabrs_over, correct_label, y_train, k, m_removal):

    counter_k_m = getLabelCnt(nabrs_over, y_train, k+m_removal)
    
    # calculate best label (as correct as possible)
    needed_cnt = 0
    for label, cnt in counter_k_m.items():
        if label != correct_label and cnt >= counter_k_m[correct_label]:
            needed_cnt += (cnt - counter_k_m[correct_label] + (label < correct_label))
        if needed_cnt <= m_removal:
            best_predict = correct_label
        else:
            # return any wrong label
            best_predict = -1
            
    # calculate worst label (as incorrect as possible)
    counter_k = getLabelCnt(nabrs_over[0:k], y_train, k)
    if getFreqLabel(counter_k) != correct_label or canGetAnotherLabel(counter_k, m_removal):
        worst_predict = -1
    else:
        worst_predict = correct_label
    
    return [worst_predict, best_predict]
    
    

def isRobust(kset, nabrs_maxk, x, y_train, m_removal, k_step):
    labels = set()
    for k in kset:
        counter_k = getLabelCnt(nabrs_maxk[0:k], y_train, k)
        labels.add(getFreqLabel(counter_k))
        if canGetAnotherLabel(counter_k, m_removal):
            return False
    return len(labels) == 1
    
    

    


# =========================
# load train and test set
import sys
import numpy as np
if len(sys.argv) < 7:
    print("args = (cleanset, attackfile, m_removal, k_min, k_max, k_step, test_cnt)")
    exit()
filename = sys.argv[1]
poisonfile = sys.argv[2]
m_removal = int(sys.argv[3])
k_min = int(sys.argv[4])
k_max = int(sys.argv[5])
k_step = int(sys.argv[6])
test_cnt = int(sys.argv[7])
with open(filename, 'rb') as f:
    XTrain = np.load(f)
    yTrain = np.load(f)
    X_test = np.load(f)
with open(poisonfile, 'rb') as f:
    poison_nums = np.load(f)
    poison_features = np.load(f)
    poison_labels = np.load(f)
    
real_posion_num = poison_nums[m_removal]
XTrain = np.concatenate((XTrain, poison_features[0:real_posion_num]), axis=0)
yTrain = np.concatenate((yTrain, poison_labels[0:real_posion_num]), axis=0)


train_cnt = XTrain.shape[0]
if test_cnt == -1:
    test_cnt = len(X_test)
X_test = X_test[:test_cnt]



import datetime
starttime = datetime.datetime.now()
# tune the hyperparameter K (#neighbors) with Cross Validation

# candidate optimal KSet
kset = list(range(k_min, k_max, k_step))
label_num = np.max(yTrain) + 1


# overapproximation on prediction results
not_robust_cnt = 0
neigh = NearestNeighbors(n_neighbors = kset[-1])
neigh.fit(XTrain)
for x in X_test:
    nabrs_maxk = neigh.kneighbors([x], return_distance=False)[0]
    if isRobust(kset, nabrs_maxk, x, yTrain, m_removal, k_step) == False:
        not_robust_cnt += 1
endtime = datetime.datetime.now()
print("#train/#cnt:", train_cnt, "/", test_cnt, "remove = ", m_removal, ",fold_num = 10.")
print("#not_robust/#total = ", not_robust_cnt, "/", test_cnt)
print("avg running time = ", (endtime - starttime).seconds / test_cnt, "s")




