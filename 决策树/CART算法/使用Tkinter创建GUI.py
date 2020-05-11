# !/usr/bin/env python    
# -*-coding:utf-8 -*-

"""
# File       : 使用Tkinter创建GUI.py
# Time       ：2020/4/28 18:56 
# Author     ：Yan You Fei
# version    ：python 3.6
# Description：
"""

from tkinter import *
from 机器学习实战算法.决策树.CART算法.CART回归树算法 import *
import matplotlib
matplotlib.use('TkAgg')  #设置后端TkAgg
#将TkAgg和matplotlib链接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def plotGUIDemo():
    root = Tk()  # 会出现一个小窗口或者一些错误提示
    myLabel = Label(root, text='Hello World')  # 在窗口上显示一些文字
    myLabel.grid()
    root.mainloop()  # 启动时间循环，使该窗口在众多时间中可以响应鼠标点击、按钮和重绘等动作

def reDraw(tolS,tolN):
    """构建树，并结果绘制出来"""
    reDraw.f.clf()  # 清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)  # 重新添加一个新图
    # 检查复选框是否被选中，如果被选中，则狗年模型树，否则构建回归树
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = createTree(reDraw.rawDat, modelLeaf, modelErr, (tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testDat, modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0].flatten().A[0],reDraw.rawDat[:,1].flatten().A[0],s=5) # 绘制散点图
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)  # 绘制连续曲线
    reDraw.canvas.draw()  # python3.6中用draw()，用show()会有警告提示

def getInputs():
    """获取输入框的值"""
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print('Enter Integer for tolN')
        # 清除错误的输入并用默认值替换
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')

    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print('Enter Float for tolS')
        # 清除错误的输入并用默认值替换
        tolSentry.delete(0,END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS


def drawNewTree():
    tolN,tolS = getInputs()  # 获取输入框的值
    reDraw(tolS,tolN)   # 绘图


if __name__ == '__main__':
    # plotGUIDemo()

    # 构建树管理器界面的tkinter小部件
    root = Tk()  # 创建一个Tk类型的根部件
    # 在Tk的GUI上放置一个画布，并用.grid()来调整布局
    reDraw.f = Figure(figsize=(5, 4), dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.draw()
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    # Label()标签，在窗口上会显示一些文字,.grid()设置行和列的位置
    # rowspan和columnspan：跨行数和跨列数
    Label(root, text='tolN').grid(row=1, column=0)
    tolNentry = Entry(root)  # 创建输入框，它是一个允许单行文本输入的输入框
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    Label(root, text='tolS').grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2,
                                                          rowspan=3)

    chkBtnVar = IntVar()  # 按钮整数值
    chkBtn = Checkbutton(root, text='Model Tree',variable=chkBtnVar)  # 创建复选按钮
    chkBtn.grid(row=3, column=0, columnspan=2)

    # 初始化一些与reDraw关联的全局变量
    reDraw.rawDat = np.mat(loadDataSet('./data/sine.txt'))
    reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]),max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)
    root.mainloop()