# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:12:30 2021

@author: oguzc
"""

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.lib import scimath as SM
import matplotlib.animation as animation
np.seterr(divide='ignore', invalid='ignore')

def sphere(first,second,third):
        xx = np.linspace(-75, 75, 1501)
        yy = np.linspace(-75, 75, 1501)
        x, y = np.meshgrid(xx, yy)

        #p = -x./sqrt(r^2-(x.^2 + y.^2));
        p1 = np.power(y,2)
        p2 = np.power(x,2)
        p3 = p1 + p2
        p4 = 50 ** 2 - (p3)
        p5 = SM.sqrt(p4)


        p = -x / p5
        q = -y / p5
        mask = np.copy(p4)

        mask[mask >= 0] = 1
        mask[mask < 0] = 0

        I = np.array([first,second,third])
        I = np.transpose(I)

        albedo = -0.5
        R = (albedo*(-I[0] * p - I[1] * q + I[2])) / (SM.sqrt(1 + np.power(p,2) + np.power(q,2)))
        
        mask = np.reshape(mask, (1501,1501))
        R = np.multiply (R, mask)
        R*=-1
        E=np.copy(R)
        E[E<0]=0
        where_are_NaNs = np.isnan(E)
        E[where_are_NaNs] = 0
        E = E / np.amax(E)
        
        plt.imshow(np.real(E), 'gray')
    
sphere(0.1, 0.34, 0.98)