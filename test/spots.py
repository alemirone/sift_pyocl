import sys, os
from math import sin, cos,pi
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
there = os.path.join(here,"..","build")
# lib = [os.path.abspath(os.path.join(there,i)) for i in os.listdir(there) if "lib" in i][0]
# sys.path.insert(0, lib)
import sift
import numpy
import scipy.misc
import pylab
lena = scipy.misc.lena()
lena[:]=0
lena[100:110,100:110] = 255
s = sift.SiftPlan(template=lena, profile=True, max_workgroup_size=8)
kp = s.keypoints(lena, just_for_spots=True)
s.log_profile()
fig = pylab.figure()
sp1 = fig.add_subplot(1, 2, 1)
im = sp1.imshow(lena, cmap="gray")
sp1.set_title("OpenCL: %s keypoint" % kp.shape[0])
sp2 = fig.add_subplot(1, 2, 2)

im = sp2.imshow(lena, cmap="gray")

for i in range(kp.shape[0]):
    x = kp[i, 0]
    y = kp[i, 1]
    print (x,y)
    scale = kp[i, 2]
    for angle in np.linspace(0.0,2*pi,4,endpoint=False)+pi/4:
        sp1.annotate("", xy=(x, y), xytext=(x + scale * cos(angle), y + scale * sin(angle)), color="red",
                     arrowprops=dict(facecolor='red', edgecolor='red', width=1),)


fig.show()

raw_input()
