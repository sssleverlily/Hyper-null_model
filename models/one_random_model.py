import math
import hypernetx as hnx
import numpy as np
import random
import xgi


def calculate_entropy(G_1):
    degree_list = np.sum(xgi.adjacency_matrix(G_1).todense(), axis=1)
    entropy_sum = 0
    for node in range(len(degree_list)):
        temp_degree = degree_list[node] / sum(degree_list)
        if temp_degree != 0:
            entropy_sum += temp_degree * math.log2(temp_degree)
    return -entropy_sum


'''
超随机换边
'''


def edge_translate_random(edge_1, edge_2):
    edge_all = edge_1 + edge_2  # [1,2,4] + [1,3] = [1,2,4,1,3]
    len_all = len(edge_all)
    edges_set = set(edge_all)  # [1,2,4,3]
    for edge in edges_set:
        edge_all.remove(edge)  # [1]
    # edge_1 edge_2 ,edge_all是多的元素

    extra_length = len(edge_all)
    can_repeat = True
    # 这时候的edges_set就是没有重复的元素

    edge_1_new = []
    edge_2_new = []

    repeat_times = 0
    for extra_node in edges_set:
        if can_repeat:
            random1 = random.random()
            random2 = random.random()
            if random1 >= 1 / 2:
                edge_1_new.append(extra_node)
            if random2 >= 1 / 2:
                edge_2_new.append(extra_node)
            if random1 >= 1 / 2 and random2 >= 1 / 2:
                repeat_times += 1
            if repeat_times == extra_length:
                can_repeat = False
        else:
            random3 = random.random()
            if random3 >= 1 / 2:
                edge_1_new.append(extra_node)
            else:
                edge_2_new.append(extra_node)

    if len(edge_1_new) + len(edge_2_new) == len_all:
        Tag = True
    else:
        Tag = False
    return edge_1_new, edge_2_new, Tag


'''
保持超度不变,交换两条边
'''


def edge_translate(edge_1, edge_2):
    edge_all = edge_1 + edge_2  # [1,2,4] + [1,3] = [1,2,4,1,3]
    edges_set = set(edge_all)  # [1,2,4,3]
    for edge in edges_set:
        edge_all.remove(edge)  # [1]

    for extra_edges in edge_all:
        edges_set.remove(extra_edges)  # [2,3,4]

    # 这时候的edges_set就是没有重复的元素
    edge_1_new = []
    edge_2_new = []

    for edge in edge_all:
        edge_1_new.append(edge)
        edge_2_new.append(edge)

    for extra_node in edges_set:
        random_num = random.randint(0, 1)
        if random_num == 0:
            edge_1_new.append(extra_node)
        else:
            edge_2_new.append(extra_node)
    return edge_1_new, edge_2_new


'''
保持超边度不变,交换两条边
'''


def edge_translate_hyperdegree(edge_1, edge_2):
    edge_1_len = len(edge_1)
    edge_2_len = len(edge_2)
    edge_all = edge_1 + edge_2  # [1,2,4] + [1,3] = [1,2,4,1,3]
    edges_set = set(edge_all)  # [1,2,4,3]

    edge_1_new = []
    edge_2_new = []
    Tag = True
    times = 0
    while len(set(edge_1_new + edge_2_new)) != len(edges_set):
        if times >= 1000:  # 换不到就不换了
            edge_1_new = edge_1
            edge_2_new = edge_2
            Tag = False
            break
        edge_1_new = random.sample(edges_set, edge_1_len)
        edge_2_new = random.sample(edges_set, edge_2_len)
        times += 1

    return edge_1_new, edge_2_new, Tag


'''
[1 2 4，5]
[1 3]
[]
[]
1：2，2:1,3:1,4:1
dic不变，list长度不变

---------1.保持超度不变，边数可以变
'''


def generate_null_model(graph):
    nswap = 10 * graph.num_edges
    max_tries = 100 * graph.num_edges
    incidence_dic = graph.edges.ids
    edge_size = len(incidence_dic)
    entropy_list = []

    tn = 0  # 尝试次数
    swapcount = 0  # 有效交换次数

    while swapcount < nswap:
        if tn >= max_tries:
            e = ('尝试次数 (%s) 已超过允许的最大次数' % tn + '有效交换次数（%s)' % swapcount)
            print(e)
            break
        tn += 1
        list_edge = list(range(edge_size))
        random_1, random_2 = random.sample(list_edge, 2)
        edge_1_new, edge_2_new = edge_translate(list(incidence_dic[random_1]), list(incidence_dic[random_2]))
        if len(set(edge_1_new)) != 0 and len(set(edge_2_new)) != 0:  # 保证超边度不为0
            incidence_dic[random_1] = set(edge_1_new)
            incidence_dic[random_2] = set(edge_2_new)
            swapcount += 1
        if swapcount % 10 == 0:
            G_1 = xgi.Hypergraph(incidence_dic)
            entropy_list.append(calculate_entropy(G_1))
    return incidence_dic, entropy_list


'''
---------2.保持超边度不变
'''


def hyperedge_degree(graph):
    nswap = 10 * graph.num_edges
    max_tries = 100 * graph.num_edges
    tn = 0  # 尝试次数
    swapcount = 0  # 有效交换次数
    entropy_list = []
    incidence_dic = graph.edges.ids
    edge_size = len(incidence_dic)

    while swapcount < nswap:
        if tn >= max_tries:
            e = ('尝试次数 (%s) 已超过允许的最大次数' % tn + '有效交换次数（%s)' % swapcount)
            print(e)
            break
        tn += 1
        list_edge = list(range(edge_size))
        list_edge_new = []
        for i in list_edge:
            list_edge_new.append(i)
        random_1, random_2 = random.sample(list_edge_new, 2)
        edge_1_new, edge_2_new, Tag = edge_translate_hyperdegree(list(incidence_dic[random_1]),
                                                                 list(incidence_dic[random_2]))
        incidence_dic[random_1] = set(edge_1_new)
        incidence_dic[random_2] = set(edge_2_new)
        if Tag:
            swapcount += 1
        if swapcount % 10 == 0:
            G_1 = xgi.Hypergraph(incidence_dic)
            entropy_list.append(calculate_entropy(G_1))
    print(swapcount)
    return incidence_dic, entropy_list


'''
---------3.保持超度和超边度都不变
'''


def both_degree(graph):
    nswap = 10 * graph.num_edges
    max_tries = 100 * graph.num_edges
    if nswap > max_tries:
        raise hnx.HyperNetXError("交换次数超过允许的最大次数")
    tn = 0  # 尝试次数
    swapcount = 0  # 有效交换次数

    node_list = list(graph.nodes)
    incidence_dic = graph.edges.ids
    edge_size = len(incidence_dic)
    entropy_list = []
    edge_list = []
    for i in range(edge_size):
        edge_list.append(i)

    while swapcount < nswap:
        if tn >= max_tries:
            e = ('尝试次数 (%s) 已超过允许的最大次数' % tn + '有效交换次数（%s)' % swapcount)
            print(e)
            break
        tn += 1
        # list_edge_new = []
        # for i in edge_list:
        #     list_edge_new.append(i)
        edge1, edge2 = random.sample(edge_list, 2)
        node1, node2 = random.sample(node_list, 2)
        if incidence_dic[edge1].__contains__(node1) and incidence_dic[edge2].__contains__(node2) \
                and not incidence_dic[edge1].__contains__(node2) and not incidence_dic[edge2].__contains__(node1):
            incidence_dic[edge1].update({node2})
            incidence_dic[edge1].remove(node1)
            incidence_dic[edge2].update({node1})
            incidence_dic[edge2].remove(node2)
            swapcount += 1
        else:
            print("没换--")
        if swapcount % 10 == 0:
            G_1 = xgi.Hypergraph(incidence_dic)
            entropy_list.append(calculate_entropy(G_1))
    return incidence_dic, entropy_list


'''
---------4.随机模型
保持节点和超边数量不变
随便吧
'''


def random_null_model(graph):
    nswap = 10 * graph.num_edges
    max_tries = 100 * graph.num_edges
    if nswap > max_tries:
        raise hnx.HyperNetXError("交换次数超过允许的最大次数")
    tn = 0  # 尝试次数
    swapcount = 0  # 有效交换次数
    incidence_dic = graph.edges.ids
    edge_size = len(incidence_dic)
    entropy_list = []

    while swapcount < nswap:
        if tn >= max_tries:
            e = ('尝试次数 (%s) 已超过允许的最大次数' % tn + '有效交换次数（%s)' % swapcount)
            print(e)
            break
        tn += 1

        list_edge = list(range(edge_size))
        list_edge_new = []
        for i in list_edge:
            # 之前是 list_edge_new.append(str(i))
            list_edge_new.append(i)
        random_1, random_2 = random.sample(list_edge_new, 2)
        edge_1_new, edge_2_new, Tag = edge_translate_random(list(incidence_dic[random_1]), list(incidence_dic[random_2]))
        if Tag:
            if len(set(edge_1_new)) >= 1 and len(set(edge_2_new)) >= 1:  # 保证超边度不为0
                incidence_dic[random_1] = set(edge_1_new)
                incidence_dic[random_2] = set(edge_2_new)
                swapcount += 1
        # if swapcount % 10 == 0:
        #     G_1 = xgi.Hypergraph(incidence_dic)
        #     entropy_list.append(calculate_entropy(G_1))

    G_1 = xgi.Hypergraph(incidence_dic)
    for node in graph.nodes.ids.keys():
        if not G_1.__contains__(node):
            random_n = random.sample(list(range(edge_size)), 1)[0]
            incidence_dic[random_n].add(node)
    return incidence_dic, entropy_list


"""
2阶零模型，保证联合度分布不变
is_half 控制是不是2.25阶零模型
"""


def twok_nullmodel(graph, is_half=False):
    nswap = 10 * graph.num_edges
    max_tries = 100 * graph.num_edges
    if nswap > max_tries:
        raise hnx.HyperNetXError("交换次数超过允许的最大次数")
    tn = 0  # 尝试次数
    swapcount = 0  # 有效交换次数
    entropy_list = []
    node_list = list(graph.nodes)
    incidence_dic = graph.edges.ids
    edge_size = len(incidence_dic)
    incidence_matrix = xgi.incidence_matrix(graph).todense()

    while swapcount < nswap:
        if tn >= max_tries:
            e = ('尝试次数 (%s) 已超过允许的最大次数' % tn + '有效交换次数（%s)' % swapcount)
            print(e)
            break
        tn += 1
        # 一开始就计算一次
        cluster_coeffs_old = xgi.clustering_coefficient(graph)
        avg_cluster_coeff_old = sum(cluster_coeffs_old.values()) / len(cluster_coeffs_old)
        random_edge_list = random.sample(list(range(edge_size)), 2)
        for node1 in incidence_dic[random_edge_list[0]]:
            for node2 in incidence_dic[random_edge_list[1]]:
                if sum(node1 in edge for edge in incidence_dic.values()) == sum(
                        node2 in edge for edge in incidence_dic.values()):
                    if is_half:
                        print("2.25")
                        # 计算聚类系数
                        cluster_coeffs = xgi.clustering_coefficient(graph)
                        # 计算平均聚类系数
                        avg_cluster_coeff = sum(cluster_coeffs.values()) / len(cluster_coeffs)
                        if avg_cluster_coeff_old == avg_cluster_coeff:
                            temp_list = incidence_dic[random_edge_list[0]]
                            incidence_dic[random_edge_list[0]] = incidence_dic[random_edge_list[1]]
                            incidence_dic[random_edge_list[1]] = temp_list
                            swapcount += 1
                        elif avg_cluster_coeff_old != avg_cluster_coeff:
                            print("cluster coefficient change! no change!")
                    else:
                        temp_list = incidence_dic[random_edge_list[0]]
                        incidence_dic[random_edge_list[0]] = incidence_dic[random_edge_list[1]]
                        incidence_dic[random_edge_list[1]] = temp_list
                        swapcount += 1
        if swapcount % 10 == 0:
            G_1 = xgi.Hypergraph(incidence_dic)
            entropy_list.append(calculate_entropy(G_1))
    print(swapcount)
    return incidence_dic, entropy_list
