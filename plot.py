import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from scipy.interpolate import interp1d

# 原始数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 平滑滤波器 - 移动平均
window_size = 5
smoothed_y = uniform_filter1d(y, size=window_size)

# 插值方法 - 样条插值
interp_func = interp1d(x, y, kind='cubic')
smoothed_x = np.linspace(min(x), max(x), num=1000)
smoothed_y = interp_func(smoothed_x)

# 绘制原始数据和平滑曲线
plt.plot(x, y, label='Original')
plt.plot(x, smoothed_y[:len(x)], label='Moving Average')
plt.plot(smoothed_x, smoothed_y, label='Spline Interpolation')
plt.legend()
plt.show()
