import pandas as pd

# 读取数据
data = pd.read_csv('iterms.csv')

def bellman_ford(graph, start):
    # 初始化
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    # 进行v-1次更新操作，循环遍历
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if dist[u] + weight < dist[v]:  # 如果（起始点到当前点的相邻点v的距离）大于（起始点到当前点的距离+当前点到相邻点u的距离）
                    dist[v] = dist[u] + weight  # 则（起始点到当前点的相邻点v的距离）更新为（起始点到当前点的距离+当前点到相邻点u的距离）

    return dist

start = data['start'].unique()

# 转换为字典
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

# 获取不同起点的数据
result_0 = bellman_ford(graph, 'v0')
print(result_0)
result_1 = bellman_ford(graph, 'v1')
print(result_1)