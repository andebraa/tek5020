import numpy as np
import PIL 
import matplotlib.patches as pat 
import matplotlib.pyplot as plt 
from min_err_rate import min_err_rate

def addbox(box):
    """
    Adds a rectangle using matplotlib.patches.Rectangle

    args:
        box (list/tuple): coordinates of bottom left and top right corner of box.
    returns:
        rect (matplotlib object): box object to be added via ax
    """
    
    rect = pat.Rectangle((box[0][0], box[0][1]), width = int(abs(box[0][0] - box[1][0])),
                          height = int(abs(box[0][1] - box[1][1])), fill = False)
    
    return rect

def norm(image):
    
    
    div = np.sum(image, axis=2)
    div[div == 0] = 1
    image[:,:,0] /= div
    image[:,:,1] /= div
    image[:,:,2] /= div
    print(np.shape(image))

    return image 

def paprika():
    image_pap = plt.imread('figures/Bilde1.png')#, format = 'rgb')
    image_pap = image_pap[:,:,:-1]
    
    normalize = False
    if normalize:
        image_pap = norm(image_pap)

    def visualize_paprika_data():
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = plt.imshow(image_fold)#, origin = 'lower')
        plt.show()

    ## paprika boxes
    box1 = [[121,171],[158,351]] #tall parika  
    box2 = [[216,74], [335,171]] #red paprika
    box3 = [[219,309], [390,403]] #green paprika

    def visualize_paparika_training_data():
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = plt.imshow(image_pap)#, origin = 'lower')

        rect1 = addbox(box1)
        ax.add_patch(rect1)

        rect2 = addbox(box2)
        ax.add_patch(rect2)

        rect3 = addbox(box3)
        ax.add_patch(rect3)

        plt.show()

    #visualize_paparika_training_data()
    #extracting training data, in not the most sexy way
    tall_pap = image_pap[box1[0][1]:box1[1][1], box1[0][0]:box1[1][0]]
    red_pap = image_pap[box2[0][1]:box2[1][1], box2[0][0]:box2[1][0]] 
    green_pap = image_pap[box3[0][1]:box3[1][1], box3[0][0]:box3[1][0]]

    

    pap_res = min_err_rate(tall_pap, red_pap, green_pap, image_pap)
    
    plt.imshow(pap_res)

    plt.title('segmentation of paprika')

    plt.show()

def folders():
    #note, y axis is flipped. 
    image_fold = plt.imread('figures/Bilde2.png')#, format = 'rgb')
   
    normalize = True
    if normalize:
        image_fold = norm(image_fold)
    

    box1_f = [[261, 383], [502, 594]] #blue folder
    box2_f = [[757, 328], [906, 472]] #red folder
    box3_f = [[673, 604], [924, 825]] #floor

    def visualize_folder_training_data():

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = plt.imshow(image_fold)#, origin = 'lower')

        rect1_f = addbox(box1_f)
        ax.add_patch(rect1_f)

        rect2_f = addbox(box2_f)
        ax.add_patch(rect2_f)

        rect3_f = addbox(box3_f)
        ax.add_patch(rect3_f)

        plt.show()


    #extracting training data, in not the most sexy way
    #blue_fold = image_fold[box1_f[0][1]:box1_f[0][1], box1_f[0][0]:box1_f[1][0]]
    #red_fold = image_fold[box2_f[0][1]:box2_f[0][1], box2_f[0][0]:box2_f[1][0]]
    #floor = image_fold[box3_f[0][1]:box3_f[0][1], box3_f[0][0]:box3_f[1][0]]

    blue_fold = image_fold[box1_f[0][1]:box1_f[1][1], box1_f[0][0]:box1_f[1][0]]
    red_fold = image_fold[box2_f[0][1]:box2_f[1][1], box2_f[0][0]:box2_f[1][0]]
    floor = image_fold[box3_f[0][1]:box3_f[1][1], box3_f[0][0]:box3_f[1][0]]

    print(box1_f[0][1])
    print(box1_f[1][1])

    print(blue_fold)
    print(red_fold)
    print(floor)

    def visualize_data():
        ax = fig.add_subplot(2,2,2)
        plt.imshow(blue_fold)

        ax = fig.add_subplot(2,2,3)
        plt.imshow(red_fold)

        ax = fig.add_subplot(2,2,4)
        plt.imshow(floor)

        plt.show()

    visualize_folder_training_data()
    print('------------')
    print(blue_fold)
    print(np.shape(blue_fold)) 
    print(np.shape(red_fold)) 
    print(np.shape(floor)) 
    fold_res =  min_err_rate(blue_fold, red_fold, floor, image_fold)

    plt.imshow(fold_res)
    plt.title('segmentation of folders')
    plt.show()
if __name__ == '__main__':
    #paprika()
    folders()
