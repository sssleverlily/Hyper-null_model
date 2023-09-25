import random
# import networkx as nx
from scipy.stats import pearsonr, spearmanr, kendalltau
import hypernetx as hnx
import csv
from collections import defaultdict
import numpy as np
import pickle
import xgi
import math
import pandas as pd


def SIR_sim_lo(graph, days, beta, gamma, initial_infected, incidence_matrix, c):
    '''

    :param graph:
    :param days:
    :param beta1:
    :param gamma:
    :param initial_infected:
    :param incidence_matrix:
    :param c:  比例，如果一条超边中感染节点个数比例>c则可以开始感染
    :return:
    '''
    # days = 1000
    if isinstance(initial_infected, int):
        initial_infected = [initial_infected]

    now_I = set(initial_infected)
    S = set(graph.nodes) - now_I  # 未感染对节点，易感态节点
    I_num, R_num = 0, 0
    R_list = []  # R态个体人数
    initial_R = set()
    n_node = incidence_matrix.shape[0]
    n_hyperedge =  incidence_matrix.shape[1]

    for _ in range(days):
        new_I = set()
        new_R = set()
        # 感染
        for node in S:
            final_beta = 1
            all_infected = 0
            # neighbour_edge = incidence_matrix[node]
            hyperedge_idList = np.where(incidence_matrix[node, :] >= 1)
            for hyperedge_id in hyperedge_idList:
                infected_node = 0
                for node_id in np.where(incidence_matrix[:, hyperedge_id].T[0] >= 1)[0]:
                    if now_I.__contains__(node_id):
                        infected_node += 1

                    infected_node = min(infected_node, c)
                    all_infected += infected_node
                # if infected_node / np.sum(incidence_matrix[:, hyperedge_id]) >= c:
                #     final_beta = final_beta * (1 - beta * math.log2(infected_node))
            final_beta = math.exp(-beta * all_infected)
            if 1 - final_beta >= random.random():
                I_num += 1
                new_I.add(node)

        # 恢复
        for node in now_I:
            if random.random() < gamma:
                R_num += 1
                new_R.add(node)

        now_I = (now_I | new_I) - new_R
        S = S - new_I
        initial_R = initial_R | new_R
        R_list.append(len(initial_R))   # initial_R

        # if len(new_R) == 0:
        #     break

    return R_list[-1]   # 返回最后一个数字
    # return I_num 返回感染人数


def new_graph(G: xgi.Hypergraph):
    node_dict = {}  # 创造一个int和str的映射
    node_temp = 0
    new_G = xgi.Hypergraph()

    for node in G.nodes:
        node_dict[node] = node_temp
        node_temp += 1

    # 将映射后的网络写入文件
    for edge in G.edges.ids.values():
        new_list = []
        for node in edge:
            new_list.append(node_dict[node])
        new_G.add_edge(new_list)
    return new_G


if __name__ == '__main__':
    day_num = 30  # 到底指定运行次数后停止
    initial_infected_rate = 0.01
    repeat_times = 10
    q = 1  # 恢复率
    c = 5
    name = 'Bars-Rev'

    for choose_type in [0]:  # -1, 0, 1, 2, 3, 4, 5
        for choose_id in [0]:  # 0, 1, 2
            for initial_infected_rate in [0.02]:  # 0.005, 0.01, 0.05, 0.1
                # for beta_c in np.array(range(1, 4)) * 0.001:
                # for beta_c in np.array(range(4, 8)) * 0.001:
                # for beta_c in np.array(range(8, 11))*0.001:
                result_beta_list = []

                save_path1 = 'E:\Epycharmprojects\Hyper_null_model\Data\\New_null_model\\' + name + '\\' + name + str(
                    choose_type) + str(initial_infected_rate) +'_sir.txt'
                if choose_type == -1:
                    origin_path = 'E:\Epycharmprojects\Hyper_null_model\Data\\' + name + '.txt'
                    # G_1 = xgi.read_edgelist(origin_path, delimiter=',')
                    G_1 = xgi.read_edgelist(origin_path)
                    G_1 = new_graph(G_1)
                else:
                    with open(
                            'E:\Epycharmprojects\Hyper_null_model\Data\\New_null_model\\' + name + '\\' + name + str(
                                    choose_type) + '_k.pkl', "rb") as tf:
                        graph_dic = pickle.load(tf)

                    G_1 = xgi.Hypergraph(graph_dic)
                    G_1 = new_graph(G_1)
                for beta_c in np.array(range(500)) * 0.001:  # np.array(range(30, 50)) * 0.001
                    print('G_1 START ')
                    incidence_matrix_1 = xgi.incidence_matrix(G_1).todense()
                    adjacency_matrix_1 = incidence_matrix_1 @ incidence_matrix_1.T  # 没有减D，先放在这
                    node_num = G_1.num_nodes

                    # 随机挑选
                    if choose_id == 0:
                        sirSourse = random.sample(list(range(node_num)), int(initial_infected_rate * node_num))
                    # 按照超度排列
                    elif choose_id == 1:
                        sirSourse = random.sample(list(np.argsort(-xgi.degree_matrix(G_1))),
                                                  int(initial_infected_rate * node_num))
                    # 按照聚类系数进行排序
                    elif choose_id == 2:
                        cluster_coeffs = xgi.clustering_coefficient(G_1)
                        node_list = sorted(cluster_coeffs, key=cluster_coeffs.get, reverse=True)
                        sirSourse = random.sample(node_list, int(initial_infected_rate * node_num))
                    elif choose_id == 3:  # 度
                        degree_list = np.sum(xgi.adjacency_matrix(G_1).todense(), axis=1)
                        node_list = list(np.argsort(-degree_list))
                        sirSourse = random.sample(node_list, int(initial_infected_rate * node_num))

                    avg_degree = 0
                    avg_degree_squared = 0
                    for i in range(node_num):
                        avg_degree += adjacency_matrix_1[i, i] / node_num
                        avg_degree_squared += adjacency_matrix_1[i, i] ** 2 / node_num

                    # I_all = np.zeros(day_num)
                    print('1+start:')
                    repeat_results = 0
                    for rt in range(repeat_times):
                        # beta1 = beta_c
                        # * avg_degree / (avg_degree_squared - avg_degree)
                        lo = SIR_sim_lo(G_1, day_num, beta_c, q, sirSourse, incidence_matrix_1, c)
                        repeat_results += lo/repeat_times
                    result_beta_list.append(repeat_results)
                        # I_all = I_all + np.array(lo)

                    # print("1+:", I_all)
                with open(save_path1, 'a+') as f:
                    f.write(str(initial_infected_rate))
                    f.write(str('  '))
                    f.write(str(choose_type))
                    f.write(str('  '))
                    f.write(str(initial_infected_rate))
                    f.write(str('  '))
                    f.write(str(choose_id))
                    f.write(str('  '))
                    f.write(str(beta_c))
                    f.write('\n')
                    f.write(str(result_beta_list))
                    f.write('\n')
