import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pickle
import hypernetx as hnx
from models.one_random_model import *
import xgi


def creat_hypergraph(name):
    save_graph_path = os.path.join('..', 'Data', name + '.pkl')
    with open(save_graph_path, "rb") as tf:
        graph_dic = pickle.load(tf)
        # graph_dic = np.load(save_graph_path, allow_pickle=True)    # 输出即为Dict 类型
        G = hnx.Hypergraph(graph_dic)
    return G


def null_model(name, graph, option, generate_times):
    if option == 0:  # 随机模型
        one_k_null_dic, entropy_list = random_null_model(graph)
    if option == 1:  # 保持超度不变
        one_k_null_dic, entropy_list = generate_null_model(graph)
    if option == 2:  # 保持超边度不变
        one_k_null_dic, entropy_list = hyperedge_degree(graph)
    if option == 3:  # 保持超度和超边度不变
        one_k_null_dic, entropy_list = both_degree(graph)
    if option == 4:
        one_k_null_dic, entropy_list = twok_nullmodel(graph)
    if option == 5:
        one_k_null_dic, entropy_list = twok_nullmodel(graph, is_half=True)
    print("null_dic_size:", len(one_k_null_dic.keys()))

    save_graph_path = os.path.join('..', 'Data', 'New_null_model', name, name + str(option) + '_k1.pkl')
    with open(save_graph_path, "wb") as tf:
        pickle.dump(one_k_null_dic, tf)
    # plt.plot(list(range(len(entropy_list))), entropy_list)
    # path = os.path.join('..', 'Data', 'New_null_model', name+'_repeat', name + str(option) + '_k' + str(generate_times) +'.npy')
    # np.save(path, entropy_list)
    print("save done!")
    # plt.show()


if __name__ == '__main__':
    for name in ['iJO1366']:
        generate_times = 10
        save_graph_path = 'E:\Epycharmprojects\Hyper_null_model\Data\\' + name + '.txt'
        # with open(save_graph_path, 'rb') as f:
        #     graph_dic = pickle.load(f)
        # print("origin_dic_size:", len(graph_dic.keys()))
        # H = xgi.Hypergraph(graph_dic)
        for choice in [0]:
            # for i in range(10, 20):
            H = xgi.read_edgelist(save_graph_path, nodetype=str)  # 如果分割是空格
            # H = xgi.read_edgelist(save_graph_path, nodetype=str, delimiter=',')  # 分割是逗号
            null_model(name, H, choice, 1)
