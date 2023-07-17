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
    

def predict(nabr_indices, y_train):
    counter = Counter([y_train[index] for index in nabr_indices])
    return getFreqLabel(counter)
    


import statistics
# perform 10-fold cross validation
def over_cross_validation(XTrain, yTrain, k):

    # split train data into 10 fold
    # we guarantee the 10 folds keep the same
    fold_num = 10
    from sklearn.model_selection import KFold
    folds = KFold(fold_num)
    
    errs = []

    for ti, vi in folds.split(XTrain, yTrain):
        XT = XTrain[ti]
        XV = XTrain[vi]
        yT = yTrain[ti]
        yV = yTrain[vi]
                
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(XT)
        nabr_indices = neigh.kneighbors(XV, return_distance=False)
        
        err_cnt = 0
  
        for i in range(XV.shape[0]):
            predicted_label = predict(nabr_indices[i], yT)
            if yV[i] != predicted_label:
                err_cnt += 1

        errs.append(err_cnt / XT.shape[0])

    return statistics.mean(errs)
    
    
        
def getOptimalK(XTrain, yTrain, kset):
    mse = []
    for k in kset:
        mse.append(over_cross_validation(XTrain, yTrain, k))
    # return best k
    return kset[mse.index(min(mse))]
    



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
    


# =====================
import datetime
if __name__ == "__main__":
    XTrain, yTrain, X_test, m_removal, kset, time_limit = getParameters()
    train_cnt = XTrain.shape[0]
    test_cnt = X_test.shape[0]

    starttime = datetime.datetime.now()
    # tune the hyperparameter K (#neighbors) with Cross Validation
    K = getOptimalK(XTrain, yTrain, kset)
    endtime = datetime.datetime.now()

    print("original learning time = ", (endtime - starttime).seconds, "s")
   















