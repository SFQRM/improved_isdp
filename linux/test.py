# -*- coding:utf-8 -*-

import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import timeit as timeit
import  time
from mpl_toolkits.mplot3d import Axes3D
path = "../figure/best_balance_parameter_cv/"
# tmp =np.linspace(1,10,9)
# # tmp = np.array(tmp)
# print "tmp:",tmp
# # np.save("../figure/best_balance_parameter_cv/test",tmp,allow_pickle=True)
# tmp2=np.load("../figure/best_balance_parameter_cv/test.npy")
# print "tmp2:",tmp2
#


number_data = 1000
epsilon_array = np.loadtxt(path+str(number_data)+"real_cv_epsilon.txt")
# c_array = np.loadtxt(path+str(number_data)+"real_cv_c_array.txt",encoding='bytes', allow_pickle=True).item()

print "epsilon_array:",epsilon_array
# print "c_array:",c_array
