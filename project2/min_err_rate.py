"""
Code based on project 1 min_error_rate in training.py. Expanded to accept three classes
"""
import numpy as np



def min_err_rate(train0, train1, train2, test):

    
    features_test = np.prod(test.shape[:-1]) 
    features_train0 = np.prod(train0.shape[:-1])
    features_train1 = np.prod(train1.shape[:-1])
    features_train2 = np.prod(train2.shape[:-1])
    features_train = np.array((features_train0, features_train1, features_train2)) 

    # a prioi probability

    prio = features_train/features_test
    #print('--------prio--------')
    #print(prio)
    #forventingsvektor, maximum likelihood for the expectationvalue
    
    #print(1/features_train1)
    #print('sum')
    #print(np.sum(np.sum(train1, axis=0), axis=0))

    my = (1/features_train0 * np.sum(train0, axis=(0,1)), # can't use arrays due to varying shapes
            1/features_train1 * np.sum(train1, axis=(0,1)),
            1/features_train2 * np.sum(train2, axis=(0,1)))
    print('my0',my[0])
    
    m = train0.shape[2]
    diff0 = (train0 - my[0])
    diff1 = (train1 - my[1])
    diff2 = (train2 - my[2])

    print('diff1', diff0)
    cov0 = np.zeros((m, m))
    cov1 = np.zeros((m, m))
    cov2 = np.zeros((m, m))
    #cov = np.zeros((m,m,3))

    for i in range(train0.shape[0]):
        for j in range(train0.shape[1]):
            elem = diff0[i, j].reshape(-1, 1)
            cov0 += (elem @ elem.T)
    for i in range(train1.shape[0]):
        for j in range(train1.shape[1]):
            elem = diff1[i, j].reshape(-1, 1)
            cov1 += (elem @ elem.T)
    for i in range(train2.shape[0]):
        for j in range(train2.shape[1]):
            elem = diff2[i, j].reshape(-1, 1)
            cov2 += (elem @ elem.T)

    """
    cov = np.zeros((train1.shape[2], train1.shape[2], 3)) #three classes, and test.shape[1] is RGB

    ret = (train1 - my[0], train2 - my[1], train3 - my[2])  
    print('ret[0]',ret[0])
    #cov1 = np.zeros((test.shape[1], test.shape[1]))
    for i in range(train1.shape[0]):
        for j in range(train1.shape[1]):
            # use res[:,None] intead of reshape(-1,1)
            elem = ret[0][i,j].reshape(-1,1) 
            cov[:,:,0] += (ret[1][i, j].reshape(-1, 1) @ ret[1][i, j].reshape(-1, 1).T)

            print(elem)
            #cov[:,:,0] += (elem @ elem.T) 

    for i in range(train2.shape[0]):
        for j in range(train2.shape[1]):
            elem = ret[1][i,j].reshape(-1,1)
            cov[:,:,1] += (elem @ elem.T) 

    for i in range(train3.shape[0]):
        for j in range(train3.shape[1]):
            elem = ret[2][i,j].reshape(-1,1)
            cov[:,:,2] += (elem @ elem.T) 
    """

    cov0 /= features_train0
    cov1 /= features_train1
    cov2 /= features_train2

    print('cov1', cov0)
    #cov /= (features_train1, features_train2, features_train3) 
    #print('cov',cov[0])
    W = (-0.5*np.linalg.pinv(cov0), 
         -0.5*np.linalg.pinv(cov1),
         -0.5*np.linalg.pinv(cov2))
    print('mu1.T', my[0].T)
    print('W[0]',W[0])
    
    w = (np.linalg.inv(cov0) @ my[0].T, 
         np.linalg.inv(cov1) @ my[1].T,
         np.linalg.inv(cov2) @ my[2].T)

    print('w[0]', w[0]) 
    #w = np.linalg.inv(cov) @ my.T #check axes here
    
    W00 = -0.5*my[0] @ w[0] - 0.5* np.log(np.linalg.det(cov0)) + np.log(prio[0])
    W10 = -0.5*my[1] @ w[1] - 0.5* np.log(np.linalg.det(cov1)) + np.log(prio[1])
    W20 = -0.5*my[2] @ w[2] - 0.5* np.log(np.linalg.det(cov2)) + np.log(prio[2])
    
    Wi0 = (W00, W10, W20)
    print('Wi0[0]',Wi0[0])

    err_rate = 0
    long_pap = [0,0,150]
    red_pap = [150,0,0]
    green_pap = [0,150,0] 
    colours = [long_pap, red_pap, green_pap] #colours for different segments
    RGB = [[1, 0.8, 0.3], [1, 0.7, 0.9], [0.1, 0.8, 0.7]]

    res = np.zeros((test.shape))
    print(res.shape)
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            g0 = test[i,j]@W[0]@test[i,j].T + w[0].T@test[i,j].T + Wi0[0]
            g1 = test[i,j]@W[1]@test[i,j].T + w[1].T@test[i,j].T + Wi0[1]
            g2 = test[i,j]@W[2]@test[i,j].T + w[2].T@test[i,j].T + Wi0[2]

            res[i,j] = RGB[np.argmax(np.array((g0,g1,g2)))]
            #err_rate += res
    #err_rate /= test.shape[0]

    return res

