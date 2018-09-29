import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import math


def fix_l(img):
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE()
    l = clahe.apply(l)

    return cv2.merge([l,a,b])

red = np.loadtxt('red.csv')
red = red.astype(np.uint8)
orange = np.loadtxt('orange.csv')
orange = orange.astype(np.uint8)

# for opencv
red = red.reshape(-1,1,3)
orange = orange.reshape(-1,1,3)

# red = cv2.cvtColor(red, cv2.COLOR_RGB2LAB)
# orange = cv2.cvtColor(orange, cv2.COLOR_RGB2LAB)

mean_r = cv2.mean(red)[:3]
mean_o = cv2.mean(orange)[:3]
mean_r = np.asarray(mean_r).reshape(3,1,-1)
mean_o = np.asarray(mean_o).reshape(3,1,-1)
mean_r = mean_r.astype(np.uint8)
mean_o = mean_o.astype(np.uint8)

m,n,o = mean_r
p,q,r = mean_o
print(m,n,o)
print(p,q,r)

red = fix_l(red)
orange = fix_l(orange)

red = red.reshape(3, 1, -1)
orange = orange.reshape(3, 1, -1)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

r, g, b = red
x, y, z = orange

ax.scatter(r,g,-b, zdir='b', c= 'red')
ax.scatter(x,y,-z, zdir='z', c= 'orange')
ax.scatter(82,179,146, c= 'blue')
ax.scatter(125,165,146, c= 'black')
plt.show()
# plt.savefig("demo.png")

