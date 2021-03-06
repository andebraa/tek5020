import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from itertools import combinations as comb

def train_test_split(data):
    """
    Script for reading and splitting data from datafiles.
    returns testdata and trainingdata
    """
    testdata = data.iloc[1::2,:]#skips every other line, starting from line 1
    traindata = data.iloc[::2,:]# --||-- starting from line 0
    
    return testdata, traindata
    
def dist(a,b):
    return np.linalg.norm(a - b, axis=1)

def nearestneighbour(train, test):

    features = test[:,1:].shape[1] 

    dim1_comb = list(comb(np.arange(features), 1))
    dim2_comb = list(comb(np.arange(features), 2))
    dim3_comb = list(comb(np.arange(features), 3))
    
    dim1_comb_list = [] #append combinations here, to find best comb later with index
    dim2_comb_list = [] 
    dim3_comb_list = []
    if features > 3:
        dim4_comb = list(comb(np.arange(features), 4)) 
        dim4_err = np.zeros(len(dim4_comb))
        dim4_comb_list = []

    dim1_err = np.zeros(features)
    dim2_err = np.zeros(len(dim2_comb)) #comb returns all combinations
    dim3_err = np.zeros(len(dim3_comb)) #of a list

    
    test_ = test[:,1:]
    train_ = train[:,1:] #leaving out collumn 1


    for indx in range(test_.shape[0]):#loop over all combinations of features
        #loop over data elements
        elem = test_[indx]
        
        dim2_indx = 0
        dim3_indx = 0
        for i in range(features):
            dim1_comb_list.append([i])
            closest_indx = np.argmin(np.abs(elem[i] - train_[:,i]))
            dim1_err[i] += test[indx, 0] != train[closest_indx, 0] #not train_!
            for j_indx, j in enumerate(range(i+1, features)):#j_indx is for indexing dim2_err, which runs from 0
                dim2_comb_list.append([i,j])
                closest_indx = np.argmin(dist(elem[[i,j]], train_[:, [i,j]]))
                dim2_err[dim2_indx] += test[indx, 0] != train[closest_indx,0]
                dim2_indx += 1    
                
                for k_indx, k in enumerate(range(j+1, features)):
                    dim3_comb_list.append([i,j,k])
                    closest_indx = np.argmin(dist(elem[[i,j,k]], train_[:, [i,j,k]]))
                    dim3_err[dim3_indx] += test[indx,0] != train[closest_indx,0]
                    dim3_indx += 1
                    
                    if features > 3:
                        for l_indx, l in enumerate(range(k+1, features)):
                            dim4_comb_list.append([i,j,k,l])
                            closest_indx = np.argmin(dist(elem[[i,j,k,l]], train_[:, [i,j,k,l]]))
                            dim4_err[0] += test[indx,0] != train[closest_indx,0]

    
    dim1_err /= float(train.shape[0])
    dim2_err /= float(train.shape[0])
    dim3_err /= float(train.shape[0])
    """
    print('------------------')
    print('dim1_err: {}\n'.format(dim1_err))
    print('dim2_err: {}\n'.format(dim2_err))
    print('dim3_err: {}\n'.format(dim3_err))
    """

    if features > 3:
         
        dim4_err /= float(train.shape[0])

        #print('dim4_err: {}\n'.format(dim4_err))
        #print('------------------------\n\n')
        return (dim1_err, dim2_err, dim3_err, dim4_err), (dim1_comb_list, dim2_comb_list, dim3_comb_list, dim4_comb_list) 
    
    #print('------------------------\n\n')
    return (dim1_err, dim2_err, dim3_err), (dim1_comb_list, dim2_comb_list, dim3_comb_list)


def min_err_rate(train, test, bc):
    
    test_ = test[:, 1:]
    test_ = test_[:, bc] #best combo
    train_ = train[:,1:] #neglecting true bool for now
    train_ = train_[:,bc]
    
    idx1 = train[:, 0] == 1
    idx2 = ~idx1
    train_class1 = train_[idx1]
    train_class2 = train_[idx2]

    # a prioi probability
    prio = (len(train_class1)/len(train) ,  len(train_class2)/len(train))

    #forventingsvektor, maximum likelihood for the expectationvalue
    my = (1/len(train_class1) * np.sum(train_class1, axis=0), 1/len(train_class2) * np.sum(train_class2, axis=0))
    cov = (1/len(train_class2)*(train_class1 - my[0]).T @ (train_class1 - my[0]),
            1/len(train_class2)*(train_class2 - my[1]).T @ (train_class2 - my[1]))

    W = (-0.5*np.linalg.pinv(cov[0]), -0.5*np.linalg.pinv(cov[1]))
    w = (np.linalg.pinv(cov[0]) @ my[0].T, np.linalg.pinv(cov[1]) @ my[1].T) 

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

def plot_featurespace(filename):
    data = pd.read_csv(filename, header=None, sep='\s+')
    data = data.to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    class_1 = data[data[:,0] == 1]
    class_2 = data[data[:,0] == 2]
    ax.scatter(class_1[:,1], class_1[:,2], class_1[:,3], label ='class 1', cmap='viridis')
    ax.scatter(class_2[:,1], class_2[:,2], class_2[:,3], label ='class 2', cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def least_squares(train_, test_, bc):
    #bc: best combo 
    train = train_[:, 1:]
    train = train[:, bc]
    test = test_[:, 1:]
    test = test[:, bc]

    indx1 = train_[:, 0] == 1
    indx2 = ~indx1
    
    Y = np.c_[np.ones((train.shape[0], 1)), train]
    b = indx1 + (-1)*indx2
    
    a_ = a(Y,b)

    err_rate = 0
    for i in range(test.shape[0]):
        g_ = g(test[i], a_)
        if g_ > 0:
            class_error = test_[i, 0] != 1
        else:
            class_error = test_[i, 0] != 2
        err_rate += class_error
    err_rate /= test.shape[0]

    return err_rate
    

def a(Y,b):
    return np.linalg.pinv(Y.T @ Y) @ Y.T @ b

def g(y,a):
    y = np.append(1,y)
    return a.T.dot(y.T) 

if __name__ == '__main__':
    
        
    for i in [1,2,3]:
        data = pd.read_csv('ds-{}.txt'.format(i), header=None, sep='\s+')
        test, train = train_test_split(data)
        test = test.to_numpy()
        train = train.to_numpy()

        error,combinations = nearestneighbour(train, test) 
        
        for err, _comb in zip(error, combinations):
            
            mer = min_err_rate(train, test, _comb[np.argmin(err)])
            lst_ = least_squares(train, test, _comb[np.argmin(err)])

            print('--------------------------------------------------')
            print("file: ds-{}.txt".format(i))
            print('best combo: {}'.format(_comb[np.argmin(err)])) 
            print('nearest neigbour: {}'.format(err[np.argmin(err)])) 
            print('minimum error rate: {}'.format(mer)) 
            print('least squares error rate: {}'.format(lst_))
            print('--------------------------------------------------\n')
            
    #min_err_rate('ds-2.txt')
    #plot_featurespace('ds-2.txt')
