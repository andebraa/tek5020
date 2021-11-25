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

    # a prioi probability

    prio = features_train/features_test
    #print('--------prio--------')
    #print(prio)
    #forventingsvektor, maximum likelihood for the expectationvalue
    
    #print(1/features_train1)
    #print('sum')
    #print(np.sum(np.sum(train1, axis=0), axis=0))

    my = (1/features_train1 * np.sum(train1, axis=(0,1)), # can't use arrays due to varying shapes
            1/features_train2 * np.sum(train2, axis=(0,1)),
            1/features_train3 * np.sum(train3, axis=(0,1)))
    print('my0',my[0])
    cov = np.zeros((train1.shape[2], train1.shape[2], 3)) #three classes, and test.shape[1] is RGB

    ret = (train1 - my[0], train2 - my[1], train3 - my[2])  
    print('ret[0]',ret[0])
    #cov1 = np.zeros((test.shape[1], test.shape[1]))
    for i in range(train1.shape[0]):
        for j in range(train1.shape[1]):
            # use res[:,None] intead of reshape(-1,1)
            elem = ret[0][i,j].reshape(-1,1) 
            #print(elem.T)
            cov[:,:,0] += (elem @ elem.T) 

    for i in range(train2.shape[0]):
        for j in range(train2.shape[1]):
            elem = ret[1][i,j].reshape(-1,1)
            cov[:,:,1] += (elem @ elem.T) 

    for i in range(train3.shape[0]):
        for j in range(train3.shape[1]):
            elem = ret[2][i,j].reshape(-1,1)
            cov[:,:,2] += (elem @ elem.T) 

    cov /= (features_train1, features_train2, features_train3) 
    print('cov',cov[0])
    W = (-0.5*np.linalg.pinv(cov[0]), 
         -0.5*np.linalg.pinv(cov[1]),
         -0.5*np.linalg.pinv(cov[2]))
    print('mu1.T', my[0].T)
    print('W[0]',W[0])
    
    w = (np.linalg.inv(cov[0]) @ my[0].T, 
         np.linalg.inv(cov[1]) @ my[1].T,
         np.linalg.inv(cov[2]) @ my[2].T)

    print('w[0]', w[0]) 
    #w = np.linalg.inv(cov) @ my.T #check axes here
    
    W10 = -0.5*my[0] @ w[0] - 0.5* np.log(np.linalg.det(cov[0])) + np.log(prio[0])
    W20 = -0.5*my[1] @ w[1] - 0.5* np.log(np.linalg.det(cov[1])) + np.log(prio[1])
    W30 = -0.5*my[2] @ w[2] - 0.5* np.log(np.linalg.det(cov[2])) + np.log(prio[2])
    
    Wi0 = (W10, W20, W30)
    print('Wi0[0]',Wi0[0])

    err_rate = 0
    long_pap = [0,0,0]
    red_pap = [1,1,1]
    green_pap = [0.5,0.5,0.5] 
    colours = [long_pap, red_pap, green_pap] #colours for different segments

    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            g1 = test[i]@W[0]@test[i].T + w[0].T@test[i].T + Wi0[0]
            g2 = test[i]@W[1]@test[i].T + w[1].T@test[i].T + Wi0[1]

            if g1-g2 > 0:
                res = test[i,0] != 1
            else:
                res = test[i,0] != 2
            err_rate += res
    err_rate /= test.shape[0]

    return err_rate

