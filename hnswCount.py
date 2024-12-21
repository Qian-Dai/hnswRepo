import hnswlib
import numpy as np


# 读取 .fvecs 文件中的数据
def read_Fvecs(file_path):
    """
    读取 fvecs 文件中的向量。
    :param file_path: fvecs 文件路径
    :return: numpy 数组，每一行是一个向量
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    vectors = []
    offset = 4  # 跳过文件头部的4字节
    dims = 960  # 修改为你的数据维度
    while offset < len(data):
        vector = np.frombuffer(data, dtype=np.float32, count=dims, offset=offset)
        vectors.append(vector)
        offset += dims * 4 + 4  # 跳过当前向量数据

    return np.vstack(vectors)


# HNSW 查询方法，记录遍历的节点
def hnsw_search_with_layer_tracking(index, query, k, ef_search):
    """
    执行 HNSW 查询，并记录被遍历的节点。
    :param index: HNSW 索引
    :param query: 查询向量
    :param k: 近邻数量
    :param ef_search: ef 参数，控制搜索的精度与速度
    :return: ids (近邻的ID), distances (近邻的距离), visited_nodes (遍历的节点)
    """
    visited_nodes = set()  # 用于记录被访问的节点

    # 执行 KNN 查询，并获得结果
    ids, distances = index.knn_query(query, k=k)

    # 记录每个查询过程中访问过的所有节点
    for node_id in ids[0]:
        visited_nodes.add(node_id)

    return ids, distances, visited_nodes


# 计算重复率
def calculate_duplicate_rate(visited_nodes, query_count):
    """
    计算最后一层节点的重复率。
    :param visited_nodes: 所有查询中访问过的节点集合
    :param query_count: 总查询次数
    :return: 重复率
    """
    node_counts = {}
    for node in visited_nodes:
        node_counts[node] = node_counts.get(node, 0) + 1

    total_duplicates = sum(count > 1 for count in node_counts.values())
    repeat_percentage = (total_duplicates / len(node_counts)) * 100  # 重复节点的百分比

    return repeat_percentage, node_counts


# 示例主程序
def main():
    index = hnswlib.Index(space='l2', dim=960)  # 960维度

    # 读取数据集
    data = read_Fvecs('/Users/austindai/Downloads/gist_base.fvecs')  # 使用实际路径
    dims = data.shape[1]

    index.init_index(max_elements=data.shape[0], ef_construction=200, M=16)  # 初始化 HNSW 索引
    index.add_items(data)  # 将数据添加到索引中

    # 假设查询向量已经准备好
    queries = np.random.rand(10, dims).astype(np.float32)  # 示例查询向量
    k = 10  # 查找10个最近邻
    ef_search = 50  # 搜索精度

    visited_nodes = []  # 用于存储每次查询中访问过的节点

    for query in queries:
        ids, distances, visited = hnsw_search_with_layer_tracking(index, query, k, ef_search)
        visited_nodes.extend(visited)

    # 计算重复率
    repeat_percentage, node_counts = calculate_duplicate_rate(visited_nodes, len(queries))

    print(f"重复率: {repeat_percentage:.2f}%")
    print(f"节点出现次数: {node_counts}")


if __name__ == "__main__":
    main()