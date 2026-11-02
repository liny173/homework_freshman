import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('iterms.csv')

def Dijkstra(G, v0, INF=float('inf')):  # inf表示该结点到起始点的距离还没有一个确切的值
    dis = dict((i, INF) for i in G.keys())  # 初始化一个距离表，这个表记录着起始结点v0到图中各个点的距离
    current_node = v0  # 一开始，当前点设置为起始点v0
    dis[v0] = 0  # 初始点到自己的距离自然为0
    visited = []  # 记录已经遍历过的结点
    list_v = []   # 判断visited是否为空
    ###
    path = dict((i, []) for i in G.keys())  # 初始化一个路径表，这个表记录着起始结点到途中各个点的最短路径
    path[v0] = str(v0)  # 初始点到自己的路径自然为自己
    ###
    while len(G) > len(visited):  # 图的结点还没被遍历完时执行循环以继续遍历
        visited.append(current_node)  # 当前点正在被遍历，所以把当前点放入visited表中
        for k in G[current_node]:  # 遍历当前点的所有相邻点
            if dis[current_node] + G[current_node][k] < dis[k]:  # 如果（起始点到当前点的相邻点k的距离）大于（起始点到当前点的距离+当前点到相邻点k的距离）
                dis[k] = dis[current_node] + G[current_node][k]  # 则（起始点到当前点的相邻点k的距离）更新为（起始点到当前点的距离+当前点到相邻点k的距离）
                seq = (path[current_node], str(k))
                sym = '->'
                path[k] = sym.join(seq)  # 起始点到（当前点的相邻点k）的最短路径，以'->'来连接seq中的两个字符串
        # 接着来选下一个当前点(current_node)
        # 从剩下未遍历的点中，选取与当前点的距离最小的那个点作为下一个当前点
        new = INF
        for node in dis.keys():
            if node in visited:
                continue
            if dis[node] < new:
                new = dis[node]
                current_node = node
        # 路径不存在，返回值为-1
        for key in path:
            if path[key] == list_v:
                path[key] = -1

    return dis, path

def add(DataFrame, str): # 可行路径权值相加
    count = 0
    for i in range(len(DataFrame)):
        if DataFrame[str][i] != float('inf'):
            count += DataFrame[str][i]
    return count

def node_count(str): # 计算结点数
    node_num = (1+len(str)/2)/2
    return node_num

start = data['start'].unique()

# 图转换为字典
list_dict = []
for x in start:
    list_key = []
    list_value = []
    for i in range(len(data)):

        if data.loc[i, 'start'] == x:

            list_key.append(data.loc[i, 'end'])
            list_value.append(data.loc[i, 'weight'])

    dic = zip(list_key, list_value)
    list_dict.append(dict(dic))

graph = dict(zip(start, list_dict))

# 数据处理
# 1. 获取各个节点的路径及权值
list_dis = []
list_path = []
for i in range(6):
    dis, path = Dijkstra(graph, v0='v' + str(i))
    list_dis.append(dis)
    list_path.append(path)

# 2. 整合路径和权值
list_total = []
for i in range(6):
    list_path[i] = pd.DataFrame.from_dict(list_path[i], orient='index', columns=['path'])
    list_path[i] = list_path[i].reset_index().rename(columns={'index': 'end'})

    list_dis[i] = pd.DataFrame.from_dict(list_dis[i], orient='index', columns=['weight'])
    list_dis[i] = list_dis[i].reset_index().rename(columns={'index': 'end'})

    total = pd.merge(list_path[i], list_dis[i], on='end', how='inner')
    list_total.append(total)

# 数据分析
# 1. 各点的最短可行路径
list_totally = [0, 0, 0, 0, 0, 0]
list_min = []
for i in range(6):
    list_totally[i] = list_total[i].sort_values(by='weight', ascending=True)
    # 剔除inf
    if list_totally[i]['weight'][list_totally[i].index[1]] != float('inf'):
        list_min.append(list_totally[i]['weight'][list_totally[i].index[1]])

arr_min = np.asarray(list_min)
# 2. 各个点所有最短可行路径的和
list_sum = []
for i in range(6):
    num = add(list_total[i], 'weight')
    list_sum.append(num)

arr_sum = np.asarray(list_sum)

# 总合各点
s = arr_sum.sum()

# 3. 各个点所有最短可行路径经过的结点数的和
list_node = []

for i in range(6):
    num_1 = 0
    for j in range(6):
        if list_total[i]['path'][j] == -1:
            num_1 += 0
        else:
            num_1 += node_count(list_total[i]['path'][j])
    list_node.append(num_1)

arr_node = np.asarray(list_node)

# 总和各点
no = arr_node.sum()

# 4. 各点最短可行路径平均结点的权值
arr_avg = arr_sum/arr_node

# 总体的平均
av = s/no

# 5. 各点经历结点的情况
node_g = np.array([list_total[0]['weight'], list_total[1]['weight'], list_total[2]['weight'],
                   list_total[3]['weight'], list_total[4]['weight'], list_total[5]['weight']])