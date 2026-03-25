import numpy as np
import time
from copy import deepcopy
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')

import kahip

from utils import *

"""Outputs a valid KaHIP input from the input KaHIP graph file. Always takes into account edges weights and only them."""
def build_kahip_input(filename:str):
    with open(path(filename, 'graphs'), 'r') as file:
        header = file.readline().strip()
        header_list = header.split(sep=" ")
        n,m,mode = int(header_list[0]), int(header_list[1]), int(header_list[2])

        xadj = np.zeros(n+1, dtype=int)
        adjncy = np.zeros(2*m, dtype=int)
        vwgt = np.ones(n, dtype=int)
        adjcwgt = np.zeros(2*m, dtype=int)

        for line_number, line in enumerate(file, 0):
                line = line.strip()
                if len(line) == 0:
                    break
                line_list = line.split(sep=" ")

                if line_number == 0:
                    xadj[line_number] = 0
                else:
                    xadj[line_number] = xadj[line_number-1] + pointer
                pointer = 0
                i = 0
                while i < len(line_list):
                    if mode == 11 and i==0:
                        vwgt[line_number] = int(line_list[i])
                        i += 1
                        continue
                    adjncy[xadj[line_number]+pointer] = int(line_list[i])
                    adjcwgt[xadj[line_number]+pointer] = int(line_list[i+1])
                    pointer += 1
                    i += 2
    xadj[-1] = 2*m

    return xadj, adjncy, vwgt, adjcwgt

def make_cuts(graph_name:str, kahip_graph:str, result_name:str, n = 1000, k = 2, epsilon = 0.03, mode = 2, blocks_name:str = ""):
    """Calcule un ensemble de n coupes-(k,epsilon) à partir d'une représentation "kahip" d'un graphe avec kaffpa, puis les stocke dans un fichier dans un dossier 'cuts' créé
    automatiquement s'il n'existe pas déjà.
    'graph_name' et 'kahip_graph' dénotent le nom des fichiers sources générés par 'model_graph', et 'result_name' celui du fichier final contenant les coupes;
    'n', 'k' et 'epsilon' aux paramètres éponymes des n coupes-(k,epsilon);
    'blocks_name' est facultatif et permet de stocker un fichier contenant les partitions de noeuds en plus des ensembles d'arêtes, utiles pour les calculs de distances
    entre coupes par exemple."""
    if not os.path.exists(path('cuts')):
        os.makedirs(path('cuts'))
    G = nx.read_gml(path(graph_name, 'graphs'))

    supress_output = 0
    xadj, adjncy, vwgt, adjcwgt = build_kahip_input(kahip_graph)

    kahip_cut_list = []
    blocks_list = []
    for _ in range(n):
        seed = np.random.randint(1000000000)
        __, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,  k, epsilon, supress_output, seed, mode)
        kahip_cut = []
        for edge in G.edges:
            if blocks[int(edge[0])] != blocks[int(edge[1])]:
                kahip_cut.append((edge[0], edge[1]))
        blocks_list.append(blocks)
        kahip_cut_list.append(kahip_cut)
    write_file(kahip_cut_list, path(result_name, 'cuts'))
    if blocks_name:
        write_file(blocks_list, path(blocks_name, 'cuts'))

def get_edge_frequency(cutlist_name:str, graph_name:str, plot_name = "", data_name = "", fit:bool = False):
    """Calcule la fréquence d'apparition des arêtes d'un graphe dans un ensemble de coupes de ce graphe, et les stocke dans un .json, puis les affiche facultativement sur
    le graphe.
    'cuts_name' et 'graph_name' dénotent respectivement le nom des fichiers sources des coupes et du graphe, et 'json_name' celui du .json résultant;
    'plot_name' est facultatif et correspond au nom du plot résultant."""
    cut_list = read_file(path(cutlist_name, "cuts"))
    G = nx.read_gml(path(graph_name, "graphs"))

    frequency_dict = dict.fromkeys(list(G.edges), 0)
    for cut in cut_list:
        for edge in cut:
            frequency_dict[edge] += 1/len(cut_list)
    
    if plot_name:
        xvalues, yvalues = compute_icdf(list(frequency_dict.values()), len(cut_list))

        # a = round((yvalues[5500] - yvalues[500])/(np.log(xvalues[5500]) - np.log(xvalues[500])),3)
        # b = round((yvalues[5500]*np.log(xvalues[500]) - yvalues[500]*np.log(xvalues[5500]))/(np.log(xvalues[5500]) - np.log(xvalues[500])),3)
        plt.figure()
        plt.plot(xvalues, yvalues)
        if fit:
            plt.plot([xvalues[500], xvalues[5500]], [yvalues[500], yvalues[5500]], color='black', alpha=0.7, label=f'{a}log(x) + {b}')
        plt.xlabel('edge frequency')
        plt.ylabel('cumulative distribution of edge frequencies')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        result_path = path(plot_name)
        plt.savefig(result_path, dpi=300)
        print(f'Edge frequencies distribution saved at {result_path}.')

    if data_name:
        nfreq_dict = {}
        for key in frequency_dict.keys():
            nfreq_dict[str(key)] = frequency_dict[key]
        freq_data = {"description" : f"Frequency of appearance of each edge of {graph_name} in {cutlist_name}; content is formatted as \"(u,v)\" : frequency.",
                 "content" : nfreq_dict}
        data_path = path(data_name)
        write_json(freq_data, data_path)
        print(f'Edge frequencies distribution saved at {data_path}.')

    return frequency_dict

def plot_cuts_metrics(cuts_names:list, graph_name:str, parameter:str, plot_LCC:bool=True, alt_suffix:str=""):
    assert parameter in ["k", "imb", "n"]
    if parameter == "n":
        parameter_ = "cuts"
    else:
        parameter_ = parameter
    G = nx.read_gml(path(graph_name, "graphs"))
    weight_dict = nx.get_edge_attributes(G, 'weight')
    LCC_norm = LCC(G)
    parameters = []
    costs_list = []
    LCCs_list = []
    for cuts_name in cuts_names:
        name = cuts_name.split(sep="_")
        print(name)
        for elem in name:
            if parameter_ in elem:
                parameters.append(float(elem.replace(parameter_, '')))
        cuts = read_file(path(cuts_name, 'cuts'))
        costs = []
        LCCs = []
        for cut in cuts:
            costs.append(sum([weight_dict[edge] for edge in cut]))
            if plot_LCC:
                G_ = nx.Graph()
                G_.add_edges_from(G.edges(data=False))
                G_.remove_edges_from(cut)
                LCCs.append(LCC(G_)/LCC_norm)
        costs_list.append(costs)
        LCCs_list.append(LCCs)

    if alt_suffix:
        suffix = alt_suffix
    else:
        suffix = f"{parameter}"
    plot_stats(parameters, costs_list, labels=[parameter, "cost"], plot_name=f"costs_{suffix}.png")
    if plot_LCC:
        plot_stats(parameters, LCCs_list, labels=[parameter, "LCC size"], plot_name=f"LCC_{suffix}.png")


"""Plots the input graph with edges frequency from the input cuts. Projection is hardcoded for Paris, and graph type must be gml."""
def plot_cut_graph(graph_name:str, freq_dict_name:str, plot_name:str, mark_infinite:bool=False):
    G = nx.MultiGraph(nx.read_gml(path(graph_name, "graphs")))
    edges_to_remove = []
    for edge in G.edges:
        if str(edge[1]) == str(len(G)-1):
            edges_to_remove.append(edge)
    G.remove_edges_from(edges_to_remove)
    G.remove_node(str(len(G)-1))
    G.graph['crs'] = ox.settings.default_crs
    G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris
    if mark_infinite:
        weight_dict = nx.get_edge_attributes(G, 'weight')

    freq_dict = read_json(path(freq_dict_name))["content"]
    cmap = mpl.colormaps['Spectral']
    colors = cmap(np.linspace(0, 1, 10000))
    norm = max(list(freq_dict.values()))
    
    edge_keys = list(G.edges)
    color_dict = dict.fromkeys(edge_keys, 'gray')
    large_dict = dict.fromkeys(edge_keys, 0.5)
    alpha_dict = dict.fromkeys(edge_keys, 0.1)
    for edge in G.edges:
        if mark_infinite and weight_dict[edge] == 10000:
            color_dict[edge] = 'black'
            alpha_dict[edge] = 1
            large_dict[edge] = 2
        freq = freq_dict[f"('{edge[0]}', '{edge[1]}')"]/norm - 0.0001
        if freq > 0:
            color_dict[edge] = mcolors.rgb2hex(colors[int(10000*freq)])
            alpha_dict[edge] = 1
            large_dict[edge] = 2
    norm=plt.Normalize(vmin=0., vmax=1.)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, ax = ox.plot_graph(G, node_size=0.1, edge_color=list(color_dict.values()), edge_linewidth=list(large_dict.values()), bgcolor='white')
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    cb.set_label('edge frequency')
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

def get_freq(cuts_name:str, graph_name:str, json_name:str, plot_name:str=""):
    """Calcule la fréquence d'apparition des arêtes d'un graphe dans un ensemble de coupes de ce graphe, et les stocke dans un .json, puis les affiche facultativement sur
    le graphe.
    'cuts_name' et 'graph_name' dénotent respectivement le nom des fichiers sources des coupes et du graphe, et 'json_name' celui du .json résultant,
    'plot_name' est facultatif et correspond au nom du plot résultant."""
    assert json_name.split(sep=".")[-1] == "json"
    get_edge_frequency(cutlist_name=cuts_name,
                    graph_name=graph_name,
                    data_name=json_name)
    if plot_name:
        plot_cut_graph(graph_name=graph_name,
                    freq_dict_name=json_name,
                    plot_name=plot_name)
    
def get_cost(cuts, graph):
    """Renvoie la liste des coûts des coupes du graphe fournis en entrée.
    'cuts' et 'graph' peuvent tous les deux contenir le nom du fichier correspondant, ou alors l'objet lui-même : 'cuts' doit alors être une liste de liste (=coupes)
    de tuples (=arêtes) comme stockée par 'make_cuts', et 'graph' un Graph networkx comme stocké par 'model_graph'."""
    if type(cuts) == str:
        cuts = read_file(path(cuts), "cuts")
    if type(graph) == str:
        graph = read_file(path(graph), "graphs")
    weight_dict = nx.get_edge_attributes(graph, "weight")
    costs = []
    for cut in cuts:
        costs.append(sum([weight_dict[edge] for edge in cut]))
    return costs

def get_isolated(blocks_name:str, graph_name:str):
    """Renvoie la liste des proportions des noeuds du graphe contenus dans la partie isolée des coupes correspondantes aux blocs fournis en entrée."""
    G = nx.read_gml(path(graph_name, "graphs"))
    n = len(G.nodes)
    blocks = read_file(path(blocks_name, "cuts"))
    insides = []
    for block in blocks:
        if block[-1] == 1: # hardcoded for k=2
            inside_indexes = 1 - np.array(block)
        else:
            inside_indexes = np.array(block)
        insides.append(np.sum(inside_indexes)/n)
    return insides

if __name__ == "__main__":
    pass
    graph_name="graph_paris"
    frac = 0
    graph_type = f"_frac{frac}"
    k = 2
    epsilon = imb = 0.02
    n = n = 1000

    # # Generic cut making
    # make_cuts(graph_name=graph_name,
    #             kahip_graph="kahip_"+graph_name+graph_type,
    #             result_name=f"cuts{n}_k{k}_imb{epsilon}{graph_type}",
    #             n=n,
    #             k=k, epsilon=epsilon,
    #             blocks_name=f"blocks{n}_k{k}_imb{epsilon}{graph_type}",
    #             )


    # # Edge frequency
    # id = ""
    # if id:
    #     cut = [read_file(path(f"cuts{n}_k{k}_imb{epsilon}{graph_type}","cuts"))[int(id.split(sep="_")[1])]]
    #     G = nx.read_gml(path(f"{graph_name}{graph_type}", "graphs"))
    #     w_d = nx.get_edge_attributes(G, "weight")
    #     cost = 0
    #     for edge in cut[0]:
    #         cost += w_d[edge]
    #     print(cost)
    #     write_file(cut, path(f"cuts{n}_k{k}_imb{epsilon}{graph_type}{id}","cuts"))
    # get_edge_frequency(cutlist_name=f"cuts{n}_k{k}_imb{epsilon}{graph_type}{id}",
    #                 graph_name=graph_name,
    #                 data_name=f"edge_freq_cuts{n}_k{k}_imb{epsilon}{graph_type}{id}.json",
    #                 #    plot_name=f"edge_freq_cuts{n}_k{k}_imb{epsilon}{graph_type}{id}.png",
    #                 )
    # plot_cut_graph(graph_name=graph_name,
    #             freq_dict_name=f"edge_freq_cuts{n}_k{k}_imb{epsilon}{graph_type}{id}.json",
    #             plot_name=f"paris_edge_freq_cuts{n}_k{k}_imb{epsilon}{graph_type}{id}.png",
    #             mark_infinite=True)

    # Cuts costs and LCC for imb
    names_list = []
    for fname in [f for f in os.listdir(path("cuts"))]:
        if "cuts" in fname and f"frac{frac}" == fname.split(sep="_")[-1]:
            if fname.split(sep="_")[2][3:] != "0.01":
                names_list.append(fname)
    plot_cuts_metrics(cuts_names=names_list, graph_name=graph_name+graph_type, parameter='imb', alt_suffix=f"imb_k{k}{graph_type}", plot_LCC=False)

    # # for k
    # k_list = [2,3,4,5,6]
    # imb = 0.5
    # names_list = []
    # for k in k_list:
    #     names_list.append(f"cuts{n}_k{k}_imb{imb}{graph_type}")
    # plot_cuts_metrics(cuts_names=names_list, graph_name=graph_name+graph_type, parameter='k', alt_suffix=f"k_imb{imb}", plot_LCC=False)

    # # for n
    # n_list = [10,100,1000,10000,100000]
    # names_list = []
    # for n in n_list:
    #     names_list.append(f"cuts{n}_k{k}_imb{imb}{graph_type}")
    # plot_cuts_metrics(cuts_names=names_list, graph_name=graph_name+graph_type, parameter='n', alt_suffix=f"_n_k{k}_imb{imb}")

        # # Fact checking balance
    # G = nx.read_gml(path(graph_name+graph_type, "graphs"))
    # cut = read_file(path(f"cuts{n}_k{k}_imb{epsilon}{graph_type}","cuts"))[0]
    # blocks = cut_to_blocks(G, cut, k)
    # results = np.zeros(2)
    # cost = 0
    # for edge in G.edges(data=True):
    #     if (edge[0],edge[1]) in cut:
    #         cost+=edge[2]["weight"]
    # for node in G.nodes(data=True):
    #     if str(node[0]) in blocks[0]:
    #         i = 0
    #     elif str(node[0]) in blocks[1]:
    #         i = 1
    #     results[i] += node[1]["weight"]
    # print(results)
    # print(cost)