import random
import statistics
import numpy as np
from collections import Counter
# to calculate neighbors quickly
from sklearn.neighbors import NearestNeighbors


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# ========================
def getFreqLabel(cntr):
    [ori_label, ori_cnt] = cntr.most_common(1)[0]
    if len(cntr.most_common(2)) < 2:
        return ori_label
    for label, cnt in cntr.items():
        if label != ori_label and cnt == ori_cnt:
            ori_label = min(ori_label, label)
    return ori_label
    

def predict(nabr_indices, y_train):
    counter = Counter([y_train[index] for index in nabr_indices])
    return getFreqLabel(counter)

    
def getLabelCnt(nabrs_indices, y_train):
    targets = [y_train[i] for i in nabrs_indices]
    return Counter(targets)
    

# min_rmv_cnt : the minimal removal cnt to detect an attack
def getMinRmvCnt(nabrs_k_m, k, y_train, m_removal, label):
    counter_k = getLabelCnt(nabrs_k_m[0:k], y_train)
    if getFreqLabel(counter_k) != label:
        return 0
    counter_k_m = getLabelCnt(nabrs_k_m[0:k+m_removal], y_train)
    counter_k_m[label] -= m_removal
    freq_wrong_label = getFreqLabel(counter_k_m)
    if freq_wrong_label == label:
        return m_removal + 1
    
    # calculate min_rmv_cnt
    left = 0
    right = m_removal
    while left < right:
        mid = (left + right) // 2
        counter_k_mid = getLabelCnt(nabrs_k_m[0:k+mid], y_train)
        counter_k_mid[label] -= mid
        if getFreqLabel(counter_k_mid) != label:
            right = mid
        else:
            left = mid + 1
    min_rmv_cnt = left
    
    return min_rmv_cnt
    

# for each k, the minimal removal to make X misclassfied
def calcuConstraint(X, ori_label, nabrs_maxk_m, yTrain, kset, m_removal):
    constraints = {}
    for k in kset:
        min_rmv_cnt = getMinRmvCnt(nabrs_maxk_m[0:k+m_removal], k, yTrain, m_removal, ori_label)
        if min_rmv_cnt <= m_removal:
            constraints[k] = min_rmv_cnt
    return constraints


    
# =================
def getInfluNeigh(predicted_label, nabr_indices_k_m, y_train):
    # check whether the predicted label can be changed by removing neighbors
    counter_k_m = Counter([y_train[index] for index in nabr_indices_k_m])
    counter_k_m[predicted_label] -= m_removal
    if getFreqLabel(counter_k_m) == predicted_label:
        return []
    else:
        return [i for i in nabr_indices_k_m if y_train[i] == predicted_label]

    

import statistics
# perform 10-fold cross validation, delete remove_indices while keep the 10-fold split
def over_cross_validation(XTrain, yTrain, k, m_removal):
    influence_map_k = [[] for i in range(len(XTrain))]
    misIndices_map_k = [[] for i in range(10)]
    # split train data into 10 fold
    # we guarantee the 10 folds keep the same
    fold_num = 10
    from sklearn.model_selection import KFold
    folds = KFold(fold_num)
    
    errs = []
    fold_index = -1

    for ti, vi in folds.split(XTrain, yTrain):
        XT = XTrain[ti]
        XV = XTrain[vi]
        yT = yTrain[ti]
        yV = yTrain[vi]
        
        fold_index += 1
                
        neigh = NearestNeighbors(n_neighbors = k + m_removal)
        neigh.fit(XT)
        nabr_indices_k_m = neigh.kneighbors(XV, return_distance=False)
        
        err_cnt = 0
  
        for i in range(XV.shape[0]):
            predicted_label = predict(nabr_indices_k_m[i][0:k], yT)
            if yV[i] != predicted_label:
                err_cnt += 1
                misIndices_map_k[fold_index].append(vi[i])
            for neigh in getInfluNeigh(predicted_label, nabr_indices_k_m[i], yT):
                influence_map_k[ti[neigh]].append(vi[i])

        errs.append(err_cnt / XV.shape[0])

    return influence_map_k, misIndices_map_k, statistics.mean(errs)
    


def getOriOptK(kset, XTrain, yTrain, m_removal):
    mse = []
    # K: x --> x' which may be predicted differently
    influence_map = {}
    # K: misclassied XTrain indeices in each group
    misIndices_map = {}
    for k in kset:
        influence_map_k, misIndices_map_k, errs = over_cross_validation(XTrain, yTrain, k, m_removal)
        influence_map[k] = influence_map_k
        misIndices_map[k] = misIndices_map_k
        mse.append(errs)
    # return best k
    return influence_map, misIndices_map, kset[mse.index(min(mse))]
    

# ========== update error =====================

def over_cross_validation_w_remove(XTrain, yTrain, k, remove_indices, influence_map_k, misIndices_map_k):
    fold_num = 10
    from sklearn.model_selection import KFold
    folds = KFold(fold_num)
    
    errs = []
    fold_index = -1
    
    toUpdate = set()
    for index in remove_indices:
        for inflenced_index in influence_map_k[index]:
            toUpdate.add(inflenced_index)

    for ti, vi in folds.split(XTrain, yTrain):
        ori_vi = vi
        
        #remove data with remove_indices
        for index in remove_indices:
            ti = ti[ti != index]
            vi = vi[vi != index]
        
        XT = XTrain[ti]
        XV = XTrain[vi]
        yT = yTrain[ti]
        yV = yTrain[vi]
        
        fold_index += 1
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(XT)
        
        err_cnt = len(misIndices_map_k[fold_index])
        
        for index in remove_indices:
            if index in ori_vi and index in misIndices_map_k[fold_index]:
                err_cnt -= 1
                
        for index in toUpdate:
            if index in vi:
                nabr_indices_k = neigh.kneighbors([XTrain[index]], return_distance=False)[0]
                predicted_label = predict(nabr_indices_k, yT)
                if yTrain[index] != predicted_label and index not in misIndices_map_k[fold_index]:
                    err_cnt += 1
                elif yTrain[index] == predicted_label and index in misIndices_map_k[fold_index]:
                    err_cnt -= 1

        errs.append(err_cnt / XV.shape[0])

    return statistics.mean(errs)
    


def updateOptK(kset, XTrain, yTrain, remove_indices, influence_map, misIndices_map):
    mse = []
    for k in kset:
        errs = over_cross_validation_w_remove(XTrain, yTrain, k, remove_indices, influence_map[k], misIndices_map[k])
        mse.append(errs)
    # return best k
    return kset[mse.index(min(mse))]

# =================


def checkAttack(assumed_Ks, x_test, ori_neigh, remove_indices, X_train, y_train, kset, influence_map, misIndices_map, ori_label):
    K = updateOptK(kset, X_train, y_train, remove_indices, influence_map, misIndices_map)
    new_X_train = np.delete(X_train, remove_indices, 0)
    new_Y_train = np.delete(y_train, remove_indices, 0)
    neigh = NearestNeighbors(n_neighbors = K)
    neigh.fit(new_X_train)
    nabr_indices = neigh.kneighbors([x_test], return_distance=False)[0]
    new_label = predict(nabr_indices, new_Y_train)
    if new_label != ori_label:
        return True
    else:
        return False



import itertools
import sys
import time
from combination import combinationIterator
#import timeout_decorator
import signal
# =========================
def findAttack(ori_neigh, x_test, ori_label, ori_K, constraint, m_removal, kset, train_cnt, X_train, y_train, influence_map, misIndices_map):
    indices_set = set()
    global attack
    ori_neigh_indices_maxk_m = ori_neigh.kneighbors([x_test], return_distance=False)[0]
    # most possible situation
    if ori_K in constraint:
        min_remove_cnt = constraint[ori_K]
        most_possible_remove_indices = [i for i in ori_neigh_indices_maxk_m[:ori_K+min_remove_cnt] if y_train[i] == ori_label][:min_remove_cnt]
        if checkAttack([ori_K], x_test, ori_neigh, most_possible_remove_indices, X_train, y_train, kset, influence_map, misIndices_map, ori_label):
            attack = most_possible_remove_indices
            return
            
 
    # second most possible situations
    for k in constraint.keys():
        min_remove_cnt = constraint[k]
        remove_candidate_indices = [i for i in ori_neigh_indices_maxk_m[:k+min_remove_cnt] if y_train[i] == ori_label]
    
        cI1 = combinationIterator(remove_candidate_indices, min_remove_cnt)
        while cI1.hasNext():
            indices1 = cI1.getNext()
            remain_candidates = [j for j in range(train_cnt) if j not in ori_neigh_indices_maxk_m[:k+min_remove_cnt]]
            for remain_remain_cnt in range(0, m_removal-min_remove_cnt+1):
                cI2 = combinationIterator(remain_candidates, remain_remain_cnt)
                while cI2.hasNext():
                    indices2 = cI2.getNext()
                    indices = indices1 + indices2
                    indices.sort()
                    if str(indices) in indices_set:
                        continue
                    if len(indices) > 0 and checkAttack([k], x_test, ori_neigh, indices, X_train, y_train, kset, influence_map, misIndices_map, ori_label):
                        attack = indices
                        return
                    if len(indices) > 0 and len(indices_set) <= 5000:
                        indices_set.add(str(indices))
    
    # situation prunning: impossible to find attacks
    impossible = True
    for k in constraint.keys():
        if constraint[k] < min_remove_cnt:
            impossible = False
    if impossible == True:
        return
    
    
    # other situations
    remove_cnt = m_removal
    remove_candidates = [j for j in range(train_cnt)]

    cI1 = combinationIterator(remove_candidates, min_remove_cnt)
    while cI1.hasNext():
        indices = cI1.getNext()
        indices.sort()
        if str(indices) in indices_set:
            continue
        if len(indices) > 0 and checkAttack([k], x_test, ori_neigh, indices, X_train, y_train, kset, influence_map, misIndices_map, ori_label):
            attack = indices
            return
        if len(indices) > 0 and len(indices_set) <= 5000:
            indices_set.add(str(indices))

# =========================
# load train and test set
import sys
import numpy as np
from ast import literal_eval
def getParameters():
    if len(sys.argv) < 8:
        print("args = (cleanset, attackfile, m_removal, k_min, k_max, k_step, test_cnt, time_limit)")
        exit()
    filename = sys.argv[1]
    poisonfile = sys.argv[2]
    m_removal = int(sys.argv[3])
    k_min = int(sys.argv[4])
    k_max = int(sys.argv[5])
    k_step = int(sys.argv[6])
    test_cnt = int(sys.argv[7])
    time_limit = int(sys.argv[8])

    
    kset = list(range(k_min, k_max, k_step))
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
    
    if test_cnt == -1:
        test_cnt = len(X_test)

    return XTrain, yTrain, X_test, kset, m_removal, time_limit, test_cnt

    

# call back when time out
def myHandler(signum, frame):
    raise Exception("Time Out Error!")

    
# =========================
def quickRobust(kset, nabrs_maxk_m, y_train, m_removal):
    labels = set()
    for k in kset:
        label = getFreqLabel(getLabelCnt(nabrs_maxk_m[0:k], y_train))
        labels.add(label)
        if len(labels) > 1:
            return False
        counter_k_m = getLabelCnt(nabrs_maxk_m[0:k+m_removal], y_train)
        counter_k_m[label] -= m_removal
        if getFreqLabel(counter_k_m) != label:
            return False
    return True
    



# =========================
def getMean(L):
    if len(L) == 0:
        return "--"
    else:
        return str(statistics.mean(L))
    
import datetime
if __name__ == "__main__":
    XTrain, yTrain, X_test, kset, m_removal, time_limit, test_cnt = getParameters()
    train_cnt = XTrain.shape[0]
    
    X_test = X_test[:test_cnt]
        
    starttime1 = datetime.datetime.now()
    
    influence_map, misIndices_map, ori_K = getOriOptK(kset, XTrain, yTrain, m_removal)
    
    endtime1 = datetime.datetime.now()
    
    unknown_indices = []
    unknown_time = []
    quick_robust_indices = []
    quick_robust_time = []
    slow_robust_indices = []
    slow_robust_time = []
    attack_indices = []
    attack_time = []
    
    neigh = NearestNeighbors(n_neighbors = np.max(kset) + m_removal)
    neigh.fit(XTrain)
    
    
    
    for index in range(len(X_test)):
        starttime2 = datetime.datetime.now()
        x = X_test[index]
        nabrs_maxk_m = neigh.kneighbors([x], return_distance=False)[0]
        ori_label = predict(nabrs_maxk_m[0:ori_K], yTrain)
            
        if quickRobust(kset, nabrs_maxk_m, yTrain, m_removal):
            endtime2 = datetime.datetime.now()
            quick_robust_indices.append(index)
            quick_robust_time.append((endtime2 - starttime2).seconds)
            
            
        else:
            global attack
            attack = []
            constraint = calcuConstraint(x, ori_label, nabrs_maxk_m, yTrain, kset, m_removal)
            timeout = False

            # not empty
            if constraint:
                # calculate attacks
                try:
                    signal.signal(signal.SIGALRM, myHandler)
                    #Execute the function myHandler when the timeout happens
                    signal.alarm(time_limit)
                    findAttack(neigh, x, ori_label, ori_K, constraint, m_removal, kset, train_cnt, XTrain, yTrain, influence_map, misIndices_map)
                except Exception:
                    timeout = True
                    print("Time Out Error!")
                    
            endtime2 = datetime.datetime.now()
            running_time = (endtime2 - starttime2).seconds

            if len(attack) > 0 :
                attack_indices.append(index)
                attack_time.append(running_time)
            elif timeout:
                unknown_indices.append(index)
                unknown_time.append(running_time)
            else:
                slow_robust_indices.append(index)
                slow_robust_time.append(running_time)
           
            
    print("ori_k=", ori_K)
    print("#train/#test:", train_cnt, "/", test_cnt, "poison_threshold = ", m_removal, ",fold_num = 10.")
    print("knn_learn_init_time =", (endtime1 - starttime1).seconds)
    print("attack% = ", len(attack_time)/test_cnt, ",avg time = ", getMean(attack_time))
    print("unknown% = ", len(unknown_time)/test_cnt, ",avg time = ", getMean(unknown_time))
    print("quick-robust%= ", len(quick_robust_time)/test_cnt, ",avg time = ", getMean(quick_robust_time))
    print("slow-robust% = ", len(slow_robust_time)/test_cnt, ",avg time = ", getMean(slow_robust_time))
    print("attack indices =", attack_indices)
    print("unknown indices =", unknown_indices)
    print("quick-robust indices= ", quick_robust_indices)
    print("slow-robust indices= ", slow_robust_indices)



    






