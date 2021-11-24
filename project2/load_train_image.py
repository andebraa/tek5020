import numpy as np
import PIL 
import matplotlib.patches as pat 
import matplotlib.pyplot as plt 
from min_err_rate import min_err_rate

image = plt.imread('figures/Bilde1.png', format = 'rgb')
image = image[:,:,:-1]
def addbox(box):
    
    
    rect = pat.Rectangle((box[0][0], box[0][1]), width = int(abs(box[0][0] - box[1][0])),
                     height = int(abs(box[0][1] - box[1][1])), fill = False)
    
    return rect


#print(image.shape) 

#image = np.flip(image, axis=0)
"""
box1 = [[121,171],[158,351]] #tall parika  
box2 = [[216,74], [335,171]] #red paprika
box3 = [[219,309], [390,403]] #green paprika


fig = plt.figure()
ax = fig.add_subplot(2,2,1)
imgplot = plt.imshow(image)#, origin = 'lower')

rect1 = addbox(box1)
ax.add_patch(rect1)

rect2 = addbox(box2)
ax.add_patch(rect2)

rect3 = addbox(box3)
ax.add_patch(rect3)


width = int(abs(box1[0][0] - box1[1][0]))
height = int(abs(box1[0][1] - box1[1][1]))

#extracting training data, in not the most sexy way
tall_pap = image[box1[0][1]:box1[1][1], box1[0][0]:box1[1][0]]
red_pap = image[box2[0][1]:box2[1][1], box2[0][0]:box2[1][0]] 
green_pap = image[box3[0][1]:box3[1][1], box3[0][0]:box3[1][0]]
"""

tall_pap= image[200:420, 100:150]
red_pap = image[100:175, 200:350]
green_pap = image[230:360, 300:420]

def visualize_data():
    ax = fig.add_subplot(2,2,2)
    plt.imshow(tall_pap)

    ax = fig.add_subplot(2,2,3)
    plt.imshow(red_pap)

    ax = fig.add_subplot(2,2,4)
    plt.imshow(green_pap)

    plt.show()


seg = min_err_rate(tall_pap, red_pap, green_pap, image) 

