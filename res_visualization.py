
# matplotlib绘制图片汉字不能正常显示问题
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path=f'./save_model/logs_resnet34_msa.csv'

data=pd.read_csv(path)
print(data)

x = data[0]
y = data["train_loss"]
# 设置图像窗口大小
plt.figure(figsize=(12, 8), dpi=80)
plt.plot(x, y)
# 数字和字符串一一对应, 数据的长度一样, ratation旋转的度数
plt.xticks(x[::3], x[::3], rotation=90)

# labelpad    Spacing in points between the label and the x-axis
plt.xlabel(u"epoch", labelpad=10)
plt.ylabel(u"loss", labelpad=10)
plt.title(u"loss变化")
plt.show()