import numpy as np
from collections import Counter
# to calculate neighbors quickly
from sklearn.neighbors import NearestNeighbors


def getFreqLabel(cntr):
    [ori_label, ori_cnt] = cntr.most_common(1)[0]
    if len(cntr.most_common(2)) < 2:
        return ori_label
    for label, cnt in cntr.items():
        if label != ori_label and cnt == ori_cnt:
            ori_label = min(ori_label, label)
    return ori_label
    

def predict(neigh, y_train, x_test):
    nabr_indices = neigh.kneighbors([x_test], return_distance=False)[0]
    targets = [y_train[index] for index in nabr_indices]
    counter = Counter(targets)
    return getFreqLabel(counter)
    


import statistics
# perform 10-fold cross validation
def over_cross_validation(XTrain, yTrain, remove_indices, k):

    # split train data into 10 fold
    # we guarantee the 10 folds keep the same
    fold_num = 10
    from sklearn.model_selection import KFold
    folds = KFold(fold_num)
    
    errs = []

    for ti, vi in folds.split(XTrain, yTrain):
        #remove data with remove_indices
        for index in remove_indices:
            ti = ti[ti != index]
            vi = vi[vi != index]
        
        XT = XTrain[ti]
        XV = XTrain[vi]
        yT = yTrain[ti]
        yV = yTrain[vi]
        
                
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(XT)
        
        err_cnt = 0
  
        for i in range(XV.shape[0]):
            predicted_label = predict(neigh, yT, XV[i])
            err_cnt += yV[i] != predicted_label

        errs.append(err_cnt / XT.shape[0])

    return statistics.mean(errs)
    
    
        
def getOptimalK(XTrain, yTrain, remove_indices, kset):
    mse = []
    for k in kset:
        mse.append(over_cross_validation(XTrain, yTrain, remove_indices, k))
    # return best k
    return kset[mse.index(min(mse))]
    
    
# =====================
from combination import combinationIterator
def find_attack(x, ori_label, m_removal, XTrain, yTrain, kset):
    for remove_cnt in range(1, m_removal + 1):
        cI = combinationIterator([j for j in range(train_cnt)], remove_cnt)
        while cI.hasNext():
            indices = cI.getNext()
            # we guarantee the same k_folds for all removal
            K = getOptimalK(XTrain, yTrain, indices, kset)
            new_X_train = np.delete(XTrain, indices, 0)
            new_Y_train = np.delete(yTrain, indices, 0)
            neigh = NearestNeighbors(n_neighbors = K)
            neigh.fit(new_X_train)
            if predict(neigh, new_Y_train, x) != ori_label:
                global attack
                attack = indices
                return



# =========================
# load train and test set
import sys
import numpy as np
def getParameters():
    if len(sys.argv) < 7:
        print("seven args = (cleanset, attackfile, m_removal, k_min, k_max, k_step, time_limit)")
        exit()
    filename = sys.argv[1]
    poisonfile = sys.argv[2]
    m_removal = int(sys.argv[3])
    k_min = int(sys.argv[4])
    k_max = int(sys.argv[5])
    k_step = int(sys.argv[6])
    time_limit = int(sys.argv[7])
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

    kset = list(range(k_min, k_max, k_step))

    return XTrain, yTrain, X_test, m_removal, kset, time_limit
    
import signal
def myHandler(signum, frame):
    raise Exception("Time Out Error!")
    
import statistics
def getMean(L):
    if len(L) == 0:
        return "--"
    else:
        return str(statistics.mean(L))


# =====================
import datetime
if __name__ == "__main__":
    XTrain, yTrain, X_test, m_removal, kset, time_limit = getParameters()
    train_cnt = XTrain.shape[0]
    test_cnt = X_test.shape[0]

    starttime1 = datetime.datetime.now()
    # tune the hyperparameter K (#neighbors) with Cross Validation
    K = getOptimalK(XTrain, yTrain, [], kset)
    neigh = NearestNeighbors(n_neighbors = K)
    neigh.fit(XTrain)
    endtime1 = datetime.datetime.now()


    unknown_time = []
    robust_time = []
    attack_time = []
    
    for index in range(test_cnt):
        global attack
        attack = []
        timeout = False
        starttime2 = datetime.datetime.now()
        x = X_test[index]
        ori_label = predict(neigh, yTrain, x)
        try:
            signal.signal(signal.SIGALRM, myHandler)
            #Execute the function myHandler when the timeout happens
            signal.alarm(time_limit - (endtime1 - starttime1).seconds)
            find_attack(x, ori_label, m_removal, XTrain, yTrain, kset)
        except Exception:
            timeout = True
            print("Time Out Error!")
       
        endtime2 = datetime.datetime.now()
        running_time = (endtime1 - starttime1 + endtime2 - starttime2).seconds
        
        if len(attack) > 0 :
            print("Test Data", index, "is not robust, an attack is", attack)
            attack_time.append(running_time)
        elif timeout:
            print("Test Data", index, "is unknwon")
            unknown_time.append(running_time)
        else:
            print("Test Data", index, "is robust")
            robust_time.append(running_time)
       
        print("running time = ", running_time, "s")
        print("----")
            
            
    print("#train/#cnt:", train_cnt, "/", test_cnt, "poison_threshold = ", m_removal, ",fold_num = 10.")
    print("attack% = ", len(attack_time)/test_cnt, ",avg time = ", getMean(attack_time))
    print("unknown% = ", len(unknown_time)/test_cnt, ",avg time = ", getMean(unknown_time))
    print("robust% = ", len(robust_time)/test_cnt, ",avg time = ", getMean(robust_time))
    
















