import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from min_err_rate import min_err_rate

# convert image to array with RGB values
img = plt.imread('figures/Bilde1.png')
img = img[:, :, :3]

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(img)

# Create a Rectangle patch
rect1 = patches.Rectangle((100, 200), 50, 220, linewidth=1, edgecolor='b',
                          facecolor='none')
rect2 = patches.Rectangle((200, 100), 150, 75, linewidth=1, edgecolor='b',
                          facecolor='none')
rect3 = patches.Rectangle((230, 300), 130, 120, linewidth=1, edgecolor='b',
                          facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

plt.axis('off')
plt.title('Markerte treningsområder', size=16)
#plt.show()

# create train data
chili = img[200:420, 100:150]
red = img[100:175, 200:350]
green = img[230:360, 300:420]
shadow = img[420:470, 155:205]

# classify and plot segmentation without normalization
segmentation = min_err_rate(img, chili, red, green)
"""
plt.imshow(segmentation)
plt.axis('off')
plt.title('Segmentering uten normalisering', size=16)
plt.show()

# normalize RGB values
sum = np.sum(img, axis=2)
sum[sum == 0] = 1
img[:, :, 0] /= sum
img[:, :, 1] /= sum
img[:, :, 2] /= sum

# plot normalized image
plt.imshow(img)
plt.axis('off')
plt.title('Normaliserte RGB-verdier', size=16)
plt.show()

# create train data of normalized image
img = img[:, :, :2]
chili = img[200:420, 100:150]
red = img[100:175, 200:350]
green = img[230:360, 300:420]

# classify and plot segmentation with normalization
segmentation = min_err_rate(img, chili, red, green)
plt.imshow(segmentation)
plt.axis('off')
plt.title('Segmentering med normalisering', size=16)
plt.show()
"""
