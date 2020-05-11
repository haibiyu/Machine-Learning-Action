# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 绘制sigmoid函数图和单位阶跃函数图.py
# Time       ：2020/4/16 17:01 
# Author     ：haibiyu
# version    ：python 3.6
# Description：
"""
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax= fig.add_subplot(111)
# 设置轴的位置
ax.spines['left'].set_position('center')
# 设置轴的颜色
ax.spines['right'].set_color('none')
# 设置轴的颜色
ax.spines['top'].set_color('none')
x= np.arange(-10,11,0.1)
y=1/(np.exp(-x)+1)
plt.plot(x,y,linewidth=3)
y1=[0 for i in x[:101]]
y2=[1 for i in x[-101:]]
plt.plot(x[:101],y1,c='red',linewidth=4)  # linewidth:线宽
plt.plot(x[-101:],y2,c='red',linewidth=4)
plt.scatter(0,0.5,c='red')
plt.xlim([-11,11])
plt.ylim([0,1])
plt.show()