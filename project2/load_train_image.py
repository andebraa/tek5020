import numpy as np
import PIL 
import matplotlib.patches as pat 
import matplotlib.pyplot as plt 

image = plt.imread('figures/Bilde1.png', format = 'rgb')

def addbox(box):
    
    
    rect = pat.Rectangle((box[0][0], box[1][1]), width = int(abs(box[0][0] - box[1][0])),
                     height = int(abs(box[0][1] - box[1][1])), fill = False)
    
    return rect


print(image.shape) 

box1 = [[110, 356],[155,170]]
box2 = [[222, 411], [384,308]]
box3 = [[215, 169], [330, 80]]


fig, ax = plt.subplots()
imgplot = plt.imshow(image)

rect1 = addbox(box1)
ax.add_patch(rect1)

rect2 = addbox(box2)
ax.add_patch(rect2)

rect3 = addbox(box3)
ax.add_patch(rect3)

#plt.show()



data1 = image[box1[0], box1[1], :-1]


