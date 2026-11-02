from 迪杰斯特拉算法及数据处理 import *

# v0为起点的输出
print(f"v0为起点，到达其他结点的最短路径:")
print(list_total[0])

print(f"-------------------------------")
# v1为起点的输出
print(f"v1为起点，到达其他结点的最短路径:")
print(list_total[1])

print(f"-------------------------------")

# 各点最短可行路径的平均值，标准差
print(f"各点最短可行路径的最小值:{arr_min.min()}")
print(f"各点最短可行路径的最大值:{arr_min.max()}")
print(f"各点最短可行路径的平均值:{arr_min.mean()}")
print(f"各点最短可行路径的方差:{round(arr_min.std(), 2)}")

print(f"-------------------------------")

# 各点最短可行路径之和的极值，平均值，标准差
print(f"各点最短可行路径之和的最小值:{arr_sum.min()}")
print(f"各点最短可行路径之和的最大值:{arr_sum.max()}")
print(f"各点最短可行路径之和的平均值:{round(arr_sum.mean(),2)}")
print(f"各点最短可行路径之和的方差:{round(arr_sum.std(), 2)}")

print(f"-------------------------------")

# 各点最短可行路径结点数之和的极值，平均值，标准差
print(f"各点最短可行路径结点数之和的最小值:{arr_node.min()}")
print(f"各点最短可行路径结点数之和的最大值:{arr_node.max()}")
print(f"各点最短可行路径结点数之和的平均值:{round(arr_node.mean(),2)}")
print(f"各点最短可行路径结点数之和的方差:{round(arr_node.std(), 2)}")

print(f"-------------------------------")

# 各点最短可行路径平均结点的权值的极值，平均值，标准差
print(f"各点最短可行路径平均结点的权值的最小值:{arr_avg.min()}")
print(f"各点最短可行路径平均结点的权值的最大值:{round(arr_avg.max(),2)}")
print(f"各点最短可行路径平均结点的权值的平均值:{round(arr_avg.mean(),2)}")
print(f"各点最短可行路径平均结点的权值的方差:{round(arr_avg.std(), 2)}")

print(f"-------------------------------")

# 总合各点的平均值
print(f"总合所有点后得到的平均权值:{round(av, 2)}")