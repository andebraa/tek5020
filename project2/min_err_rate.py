"""
Code based on project 1 min_error_rate in training.py. Expanded to accept three classes
"""
import numpy as np



def min_err_rate(train1, train2, train3, test):

    
    features_test = np.prod(test.shape[:-1]) 
    features_train1 = np.prod(train1.shape[:-1])
    features_train2 = np.prod(train2.shape[:-1])
    features_train3 = np.prod(train3.shape[:-1])
    features_train = np.array((features_train1, features_train2, features_train3)) 

    print('n',features_test)
    print('n1',features_train1)
    print('n2',features_train2)
    print('n2',features_train3)
    # a prioi probability

    print(train1.shape[0]*train1.shape[1])
    prio = features_train/features_test
    print('--------prio--------')
    print(prio)
    #forventingsvektor, maximum likelihood for the expectationvalue
    my = (1/features_train1 * np.sum(train1, axis=(0,1)), 
            1/features_train2 * np.sum(train2, axis=(0,1)),
            1/features_train3 * np.sum(train3, axis=(0,1)))
    
    cov = np.zeros((train1.shape[2], train1.shape[2], 3)) #three classes, and test.shape[1] is RGB
    print(np.shape(my)) 
    print(np.shape(my[1]))
    print(np.shape(train1[0,0] - my[1]))
    print('break')
    print(np.shape(cov))

    #cov1 = np.zeros((test.shape[1], test.shape[1]))
    for i in range(train1.shape[0]):
        for j in range(train1.shape[1]):
            # use res[:,None] intead of reshape(-1,1)
            elem = (train1[i,j] - my[1]).reshape(-1,1)
            #print(elem.shape)
            cov[:,:,1] += (elem @ elem.T) 

    for i in range(train2.shape[0]):
        for j in range(train2.shape[1]):
            elem = (train2[i,j] - my[2]).reshape(-1,1)
            cov[:,:,2] += (elem @ elem.T) 

    for i in range(train3.shape[0]):
        for j in range(train3.shape[1]):
            elem = (train3[i,j] - my[3]).reshape(-1,1)
            cov[:,:,3] += (elem @ elem.T) 

    cov /= (features_train1, features_train2, features_train3) 

    #W = (-0.5*np.linalg.pinv(cov[0]), -0.5*np.linalg.pinv(cov[1]))
    
    w = (np.linalg.pinv(cov[0]) @ my[0].T, np.linalg.pinv(cov[1]) @ my[1].T)

    w = np.linalg.inv(cov) @ my.T #check axes here
    
    W10 = -0.5*my[0] @ w[0] - 0.5* np.log(np.linalg.det(cov[0])) + np.log(prio[0])
    W20 = -0.5*my[1] @ w[1] - 0.5* np.log(np.linalg.det(cov[1])) + np.log(prio[1])
    Wi0 = (W10, W20)

    err_rate = 0
    for i in range(test_.shape[0]):
        g1 = test_[i]@W[0]@test_[i].T + w[0].T@test_[i].T + Wi0[0]
        g2 = test_[i]@W[1]@test_[i].T + w[1].T@test_[i].T + Wi0[1]

        if g1-g2 > 0:
            res = test[i,0] != 1
        else:
            res = test[i,0] != 2
        err_rate += res
    err_rate /= test_.shape[0]

    return err_rate

