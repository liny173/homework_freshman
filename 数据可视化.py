from 迪杰斯特拉算法及数据处理 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

fig_1 = plt.figure(num=1, figsize=(12, 8))

# 子图1(各个起点最短可行路径之和)
ax_1 = fig_1.add_subplot(2, 2, 1)

# 柱状图
colors = ['#476048', '#859a6d', '#99cf86', '#95bcb9', '#81b7bb', '#6e9985']
ax_1.bar(list_total[0]['end'], arr_sum, width=0.5, zorder=2, color=colors)

# y轴范围
ax_1.set_ylim(0, 160)
ax_1.yaxis.set_major_locator(MultipleLocator(20))
ax_1.yaxis.set_minor_locator(MultipleLocator(5))
ax_1.yaxis.grid(linestyle=':', alpha=0.5, which='both', zorder=-1)

# 共享x轴，绘制折线
ax_2 = ax_1.twinx()
ax_2.set_ylim(0, 160)

ax_2.yaxis.set_major_locator(MultipleLocator(20))
ax_2.yaxis.set_minor_locator(MultipleLocator(5))
ax_2.plot(list_total[0]['end'], arr_sum, linestyle='-.', marker='.', alpha=0.7, color='#6e9985',label='path sum')

# 标签
ax_1.set_xlabel("origin")
ax_1.set_ylabel("shortest path sum")
ax_1.set_title("Varying-Origin of Shortest Path Sum")
ax_2.legend()

# 值
for i, j in zip(list_total[0]['end'], arr_sum):
    ax_1.text(i, j, j, ha='center', va='bottom')

# 子图2(各个起点最短可行路径经历的结点数之和)
ax_3 = fig_1.add_subplot(2, 2, 2)

# y轴范围
ax_3.set_ylim(0, 16)
ax_3.yaxis.set_major_locator(MultipleLocator(4))
ax_3.yaxis.set_minor_locator(MultipleLocator(1))

# 共享x轴，绘制折线图
ax_4 = ax_3.twinx()
ax_4.set_ylim(0, 16)
ax_4.yaxis.set_major_locator(MultipleLocator(4))
ax_4.yaxis.set_minor_locator(MultipleLocator(1))

# 网格
ax_3.yaxis.grid(linestyle=':', alpha=0.5, which='both', zorder=-1)
ax_3.xaxis.grid(linestyle=':', alpha=0.5, zorder=-1)

# 颜色设置
color = ['#44b4e1', '#4c536f', '#96c8bc', '#63a288', '#bfe2b4', '#44c662']

# 散点图
ax_3.scatter(list_total[0]['end'], arr_node, s=arr_node, c=color, zorder=4)
ax_3.scatter(list_total[0]['end'], arr_node, s=arr_node*100, c=color, alpha=0.5, zorder=3)
ax_3.scatter(list_total[0]['end'], arr_node, s=arr_node*400, c=color, alpha=0.3, zorder=2)

ax_4.plot(list_total[0]['end'], arr_node, linestyle=':', color='#5b9185', zorder=1, alpha=0.5, label='quantity of node')

# 标签
ax_3.set_xlabel("origin")
ax_3.set_ylabel("quantity of node in shortest path")
ax_3.set_title("Varying-Origin of Quantity of Node in Shortest Path")
ax_4.legend()

# 子图3(各点最短可行路径平均结点的权值)
ax_5 = fig_1.add_subplot(2, 2, 3)
ax_5.plot(list_total[0]['end'], arr_avg, color='#189e7f', alpha=0.5, label='point weight')

# x, y轴范围
ax_5.set_ylim(0, 20)
ax_5.set_xlim(0, 5)
ax_5.yaxis.set_major_locator(MultipleLocator(4))
ax_5.yaxis.set_minor_locator(MultipleLocator(1))

# 网格
ax_5.yaxis.grid(linestyle=':', alpha=0.5, which='both', zorder=-1)

# 添加标准线
ax_5.axvline(x=2, ls=':', c='#189e7f', alpha=0.5)

# 填充
ax_5.fill_between(list_total[0]['end'], y1=0, y2=arr_avg, color='#a4e0d7', alpha=0.5)
ax_5.fill_between(list_total[0]['end'], y1=0, y2=arr_avg, where=(arr_avg > av), color='#189e7f', alpha=0.5)

# 添加文本
ax_5.text('v0', 10, '          y >= average')
ax_5.text('v2', 16, '     totally average weight = ' + str(round(av, 2)))
ax_5.text('v3', 2, 'y < average')
ax_5.legend()

# 标签
ax_5.set_title("Varying-Origin of Average Weight")
ax_5.set_xlabel("origin")
ax_5.set_ylabel("average weight")

# 子图4
ax_6 = fig_1.add_subplot(2, 2, 4)
im = ax_6.imshow(node_g, cmap='Blues')

# x, y轴的刻度标签
ax_6.set_xticks(np.arange(6), labels=list_total[0]['end'])
ax_6.set_yticks(np.arange(6), labels=list_total[0]['end'])

# 值
for i in range(6):
    for j in range(6):
        ax_6.text(j, i, node_g[i, j], ha='center', va='center')

# 标签
ax_6.set_title("Path Weight Between Two Points")
ax_6.set_xlabel("destination")
ax_6.set_ylabel("origin")

# 颜色条
plt.colorbar(im)

plt.tight_layout()
plt.show()