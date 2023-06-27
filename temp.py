# @Time : 2023/6/27 17:22
# @Author : Li Jiaqi
# @Description :
import matplotlib.pyplot as plt

a=[0,2,6,12]
b=[72.6,67.9,47.2,28.4]

plt.plot(a,b)
plt.xlabel('Frozen Layers')
plt.ylabel('Eval Dice')
plt.show()