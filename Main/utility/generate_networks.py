import random
# random.seed(1)
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import networkx as nx


def make_modular_network1(N, average_degree, m, p1, p2):
    '''
    按平均度构建网络
    :param N:
    :param average_degree:
    :param m:
    :param p1:
    :param p2:
    :return:
    '''
    assert N % m == 0, 'N must be devisible by m'

    print("MESN: m={0} p1={1} p2={2}".format(m, p1, p2))
    G = np.zeros((N, N))
    size = N/m  # 一个模块中的节点数量

    Point_indegree = [0 for i in range(N)]  # 存放各节点的入度

    for i in range(N):
        com_index = i//size  # 模块索引，从0开始
        k_in_prev = 0
        k_out_prev = 0

        for j in range(int(size*com_index)):  # 不同一个size下，模块外，出度
            if G[i][j] != 0:
                k_out_prev += 1
        for j in range(int(size*com_index), int(size*(com_index+1))):  # 同个size中，模块内，入度
            if G[i][j] != 0:
                k_in_prev += 1
        for j in range(i, N):
            if j < size*((i//size)+1):
                '''
                分子：平均度乘以模块内的连接度=一个节点的总入度，总入度-已有的入度=待分配的入度
                分母：表示处于当前模块内还有多少个节点未被访问，一个模块的总节点数size，i表示在整个N中的第i个节点
                例如，size=5,现在是第2个模块，索引为1，前面有模块0，i为7，那么分母表示5-(7-5*1)+1=4 表示当前模块内还未分配的节点个数，78910共四个
                '''
                if np.random.rand() < (average_degree * p1 - k_in_prev)/(size-(i-(size*com_index))+1):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
                    # G[i][j] = np.random.uniform(-0.5,0.5)
                    # G[j][i] = np.random.uniform(-0.5,0.5)
            else:
                '''
                分母：模块外的节点总数
                '''
                if np.random.rand() < (average_degree * p2 - k_out_prev)/(N-(size * ((i//size)+1) )+1):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
                    # G[i][j] = np.random.uniform(-0.5,0.5)
                    # G[j][i] = np.random.uniform(-0.5,0.5)


    indegree, times = count_indegree(G)
    count_edges(G)
    return G, indegree, times


def draw_network(G_array):
    path = "D:\\Pycharm\\Project\\ML\\ESN\\Image\\"

    G = nx.from_numpy_matrix(G_array)
    in_degree = G.edges(data=True)._adjdict
    # for i in in_degree:
    #     print("N={0}：{1} in_degrees".format(i, len(in_degree[i])))
    #     in_d[len(in_degree[i])] += 1
    print(G)  # Graph with 200 nodes and 1962 edges
    # print(G.degree())# 打印每个节点的度为多少

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    d = dict(nx.degree(G))
    print("图中度从大到小排序：",degree_sequence)
    print("节点总数：",len(degree_sequence))  # 结点个数
    print("节点总度：",sum(degree_sequence))
    print("平均度为：", sum(d.values()) / len(G.nodes))
    # print("图的入度：", G.in_degree())
    # plt.figure(figsize=(12, 9))
    # nx.draw_networkx(G)
    # plt.savefig(path + 'ESN2_num.png', dpi=100)

    print(G)
    # レイアウトの取得
    pos = nx.spring_layout(G)

    # 可視化
    plt.figure(figsize=(12, 9))
    nx.draw_networkx_edges(G, pos, edge_color="steelblue")
    nx.draw_networkx_nodes(G, pos, node_color="blue")
    plt.axis('on')
    # plt.savefig(path+'normal_modularity_0.4.png', dpi=100)
    # plt.savefig(path + 'ESN2_blue.png', dpi=100)
    plt.show()


def make_ESN(n, p1):
    '''
    按节点划分
    '''

    print("ESN: n={0}，p1={1}".format(n, p1))
    W = np.zeros((n, n))
    number = round(p1 * (n-1))
    for i in range(n):
        _range = list(range(0, i))+list(range(i+1, n))
        out_point = random.sample(_range, number)
        out_w = np.random.randn(1, number)
        # out_w = np.random.uniform(-1, 1, size=(1, number))
        W[i, out_point] = out_w  # 单向
        # W[out_point, i] = np.random.randn(1, number)  # 单向

    # count_edges(W)
    # indegree, times = count_indegree(W)
    return W


def set_in(m_index, p1, size, W):
    # 模块内连接
    number = round(p1 * (size - 1))

    for row in range(m_index*size, (m_index+1)*size):
        _range = list(range(m_index*size, row)) + list(range(row + 1, (m_index+1)*size))
        out_point = random.sample(_range, number)
        out_w = np.random.randn(1, number)
        # out_w = np.random.uniform(-0.1, 0.1, size=(1, number))
        W[row, out_point] = out_w

    return W


def set_out(m_index, n_index, p2, size, W):
    # 模块间连接
    number = round(p2 * size)

    for row in range(m_index * size, (m_index+1)*size):
        _range = list(range(n_index*size, (n_index+1)*size))
        out_point = random.sample(_range, number)
        out_w = np.random.randn(1, number)
        # out_w = np.random.uniform(-0.1, 0.1, size=(1, number))
        W[row, out_point] = out_w

    for row in range(n_index * size, (n_index+1)*size):
        _range = list(range(m_index*size, (m_index+1)*size))
        out_point = random.sample(_range, number)
        out_w = np.random.randn(1, number)
        # out_w = np.random.uniform(-0.1, 0.1, size=(1, number))
        W[row, out_point] = out_w

    return W


def make_modular_network_node(N, m, p1, p2):
    '''
    按照节点连接，模块内，模块间
    '''
    assert N % m == 0, 'N must be devisible by m'

    print("M-ESN: m={0} p1={1} p2={2}".format(m, p1, p2))
    W = np.zeros((N, N))  # 储层权值初始化为0
    size = int(N/m)  # 一个模块中的节点数量
    Point_num = [0 for i in range(int(N))]  # 存放不加权节点度

    # 模块内连接
    for i in range(0, m):
        W = set_in(i, p1, size, W)

    # 模块间连接
    for i in range(0, m):
        for j in range(i+1, m):
            W = set_out(i, j, p2, size, W)

    count_edges(W)
    Point_indegree = count_indegree(W)

    return W, Point_indegree



def make_small_world(N, m, p1, p2):
    e = int(p1*N/m)
    W = nx.random_graphs.barabasi_albert_graph(N, e)  # 生成一个n=1000，m=5的BA无标度网络
    # pos = nx.spring_layout(G)
    W = nx.to_numpy_array(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i][j] != 0:
                W[i][j] = np.random.randn()


    Point_num = [0 for it in range(int(N))]  # 存放不加权节点度
    val = []
    for row in range(N):
        tmp = np.count_nonzero(W[:, row])
        v = sum(W[:, row])
        val.append(v)
        Point_num[tmp] += 1

    Point_indegree = count_indegree(W)
    return W, Point_indegree


def make_esn(n, p1):
    '''
    随机构建储层
    '''
    W = np.random.rand(n, n) - 0.5  # 储备池连接矩阵 N * N
    Point_indegree = count_indegree(W)
    return W, Point_indegree


def count_indegree(w):
    indegree, times = [], []
    all_indegree = []

    for i in range(w.shape[0]):
        id = sum(w[:, i])
        # print("id", id)
        # print(id in indegree)
        # print(indegree)
        all_indegree.append(id)
        if id in indegree:
            index = indegree.index(id)
            # print("index:", index)
            times[index] += 1
        else:
            indegree.append(id)
            times.append(1)

    avg_indegree = np.mean(all_indegree)
    sd_indegree = np.std(all_indegree)
    print("Indegree mean:{0}, sd:{1}, sd/mean:{2}".format(avg_indegree, sd_indegree, sd_indegree / avg_indegree))
    return indegree, times


def count_all_edges(g):
    sum = 0
    for i in range(g.shape[0]):
        tmp = len(np.nonzero(g[:, i])[0])
        sum += tmp
    # print("总连接边数：{0}".format(sum))
    return sum


def count_edges(g):
    '''
    计算边，将双向连接的边视作一条边
    '''
    sum = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[0]):
            if g[i][j] * g[j][i] != 0:
                sum += 1
            elif g[i][j] != 0:
                sum += 2
    sum /= 2
    # print("双向连接视为一条边，总边数为：{0}".format(sum))
    return sum


def count_two_way(W):
    Num_two_way = 0
    for i in range(W.shape[0]):
        for j in range(i, W.shape[0]):
            if W[i][j] * W[j][i] != 0:
                Num_two_way += 1
    # print("Two_way_edge:{0}".format(Num_two_way))
    return Num_two_way


def count_all_two_way(W):
    Num_two_way = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i][j] * W[j][i] != 0:
                Num_two_way += 1
    # print("Two_way_edge:{0}".format(Num_two_way))
    return Num_two_way


def adjust_two_way(W, two_way_ratio):
    p1, p2 = [], []  # two_way_point
    p3, p4 = [], []  # one_way_point

    for i in range(W.shape[0]):
        for j in range(i, W.shape[0]):
            a, b = W[i][j], W[j][i]
            if W[i][j] != 0 and W[j][i] != 0:  # 双向连接
                p1.append(i)
                p2.append(j)
            elif W[i][j] == 0 and i != j:
                p3.append(i)
                p4.append(j)

    for i, j in zip(p1, p2):
        print(W[i][j], W[j][i])

    Num_aim = count_all_edges(W) * two_way_ratio
    Num_two_way = count_all_two_way(W)
    print(Num_two_way, count_all_two_way(W))
    print("Before Two_way_edge:{0}".format(round(Num_two_way / count_all_edges(W), 2)))
    x = Num_aim - Num_two_way
    print(x)

    if x < 0:  # 如果双向连接的边数比目标多，需要删去
        index = [i for i in range(len(p1))]
        print(index)
        num = (-1 * x) / (2 - two_way_ratio)
        # num = (-1 * x) / 2
        # del_edges = random.sample(index, int(Num_two_way-Num_aim))
        print(len(index), num)
        print((Num_two_way - 2 * num) / (count_all_two_way(W) - num))
        del_edges = random.sample(index, int(num))

        for i in del_edges:
            print(i)
            l, r = p1[i], p2[i]
            print(l, r, W[l][r], W[r][l])
            W[l][r] = 0


    elif x > 0:  # 如果双向连接的边数比目标少，需要添加
        index = [i for i in range(len(p3))]
        num = x / (2 - 2 * two_way_ratio)
        # print("num:", num, index)
        # add_edges = random.sample(index, int(Num_aim-Num_two_way))
        add_edges = random.sample(index, int(num))

        for i in add_edges:
            l, r = p3[i], p4[i]
            W[l][r] = np.random.randn()
            W[r][l] = np.random.randn()

    print("Now Two_way_edge:{0}".format(round(count_all_two_way(W) / count_all_edges(W), 2)))
    return W


def adjust_neg_ratio(W, neg_ratio):
    p1, p2 = [], []  # two_way_point
    p3, p4 = [], []  # one_way_point
    pos, neg = 0, 0
    Num_two_way = 0

    for i in range(W.shape[0]):
        for j in range(i, W.shape[0]):
            if W[i][j] * W[j][i] != 0:  # 双向连接
                if W[i][j] * W[j][i] > 0:  # 正反馈
                    p1.append(i)
                    p2.append(j)
                    pos += 1
                else:  # 负反馈
                    p3.append(i)
                    p4.append(j)
                    neg += 1
                Num_two_way += 1

    print("Before pos:{0} neg:{1}".format(round(pos / (pos + neg), 2), round(neg / (pos + neg), 2)))
    # print("Two_way_edge:{0}".format(Num_two_way))
    Num_aim = int(Num_two_way * neg_ratio)  # 负反馈目标数量
    Num_neg = neg  # 负反馈目前数量

    # print(Num_aim, Num_neg, len(index))
    if Num_neg < Num_aim:  # 不足
        index = [i for i in range(len(p1))]
        neg_edges = random.sample(index, int(Num_aim - Num_neg))
        for i in neg_edges:
            l, r = p1[i], p2[i]
            random_num = np.random.uniform(0, 1)
            if random_num < 0.5:
                W[l][r] *= (-1)
            else:
                W[r][l] *= (-1)
    else:  # 过多
        index = [i for i in range(len(p3))]
        neg_edges = random.sample(index, int(Num_neg - Num_aim))
        for i in neg_edges:
            l, r = p3[i], p4[i]
            random_num = np.random.uniform(0, 1)
            if random_num < 0.5:
                W[l][r] *= (-1)
            else:
                W[r][l] *= (-1)

    pos, neg = 0, 0
    for i in range(W.shape[0]):
        for j in range(i, W.shape[0]):
            if W[i][j] * W[j][i] > 0:
                pos += 1
            elif W[i][j] * W[j][i] < 0:
                neg += 1
    print("Now pos:{0} neg:{1}".format(round(pos / (pos + neg), 2), round(neg / (pos + neg), 2)))
    return W


def adjust_degree(W, n, average_degree):
    Degree_aim = int(n * average_degree)
    Degree_now = count_all_edges(W)
    print("Before total number of degree:{0}, Aim:{1}".format(Degree_now, average_degree * n))

    if Degree_now < Degree_aim:
        minus = Degree_aim - Degree_now
        while minus > 0:
            l, r = np.random.randint(0, n), np.random.randint(0, n)
            if W[l][r] == 0:
                W[l][r] = np.random.randn()
                minus -= 1
    else:
        minus = Degree_now - Degree_aim
        while minus > 0:
            l, r = np.random.randint(0, n), np.random.randint(0, n)
            if W[l][r] != 0:
                W[l][r] = 0
                minus -= 1
    print("Now total number of degree:{0}".format(Degree_now))
    return W


def make_modular_network(N, average_degree, m, p1, p2, neg_ratio=0.5, tw_ratio=0.5):
    '''
    具有反馈结构MESN
    N：储层节点
    average_degree: 平均度
    m: 模块数量
    p1: 模块内连接概率
    p2: 模块间连接概率
    neg_ratio: 负反馈比例
    tw_ratio：双向连接比例
    '''

    assert N % m == 0, 'N must be devisible by m'
    print("MESN: m={0} p1={1} p2={2} Ad={3} NegRatio={4} TwoWayRatio={5}".format(m, p1, p2,
                                                                                 average_degree, neg_ratio, tw_ratio))
    G = np.zeros((N, N))
    size = N / m  # 一个模块中的节点数量

    for i in range(N):
        com_index = i // size  # 模块索引，从0开始
        k_in_prev = 0
        k_out_prev = 0

        for j in range(int(size * com_index)):  # 不同一个size下，模块外，出度
            if G[i][j] != 0:
                k_out_prev += 1
        for j in range(int(size * com_index), int(size * (com_index + 1))):  # 同个size中，模块内，入度
            if G[i][j] != 0:
                k_in_prev += 1
        for j in range(i, N):
            if j < size * ((i // size) + 1):
                '''
                分子：平均度乘以模块内的连接度=一个节点的总入度，总入度-已有的入度=待分配的入度
                分母：表示处于当前模块内还有多少个节点未被访问，一个模块的总节点数size，i表示在整个N中的第i个节点
                例如，size=5,现在是第2个模块，索引为1，前面有模块0，i为7，那么分母表示5-(7-5*1)+1=4 表示当前模块内还未分配的节点个数，78910共四个
                '''
                if np.random.rand() < (average_degree * p1 - k_in_prev) / (size - (i - (size * com_index)) + 1):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
                    # G[i][j] = np.random.uniform(-0.5,0.5)
                    # G[j][i] = np.random.uniform(-0.5,0.5)
            else:
                '''
                分母：模块外的节点总数
                '''
                if np.random.rand() < (average_degree * p2 - k_out_prev) / (N - (size * ((i // size) + 1)) + 1):
                    G[i][j] = np.random.randn()
                    G[j][i] = np.random.randn()
                    # G[i][j] = np.random.uniform(-0.5,0.5)
                    # G[j][i] = np.random.uniform(-0.5,0.5)

    G = adjust_degree(G, N, average_degree)
    G = adjust_neg_ratio(G, neg_ratio)
    # Point_num, times = count_indegree(G)  # 存放节点度
    print("M-ESN model OK!")

    return G


def computeedge(g):
    sum = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i][j] != 0:
                sum += 1
    return sum
