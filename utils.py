import os
import pickle
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import collections
import itertools

filepath = "./data/User/workspaceStorage/"

def path(file:str="", folder:str=""):
    if file:    
        if folder:
            return os.path.join(filepath, folder, file)
        else:
            return os.path.join(filepath, file)
    else:
        return filepath

def write_json(dictionary:dict, filename:str):
    with open(filename, "w+", encoding='utf-8') as outfile:
        json.dump(dictionary, outfile, indent = 4)

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as openfile:
        json_object = json.load(openfile)
    
    return json_object

def write_file(object, file_name):
    with open(file_name, 'wb+') as f:
        pickle.dump(object, f)

def read_file(file_name):
    with open(file_name, 'rb') as f:
        object = pickle.load(f)
    return object

"""Returns a dict{node:set((neighbor(nodes), weight(node,neighbor)))} from a weighted nx.Graph."""
def find_neighbors(G, weight_dict:dict):
    nodes_neighbors_edges = {}
    for edge in G.edges:
        if edge[0] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[0]] = set()
        nodes_neighbors_edges[edge[0]].add((edge[1], weight_dict[edge]))
        if edge[1] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[1]] = set()
        nodes_neighbors_edges[edge[1]].add((edge[0], weight_dict[edge]))
    
    return nodes_neighbors_edges

"""From a simple newtorkx graph, returns a dictionnary with keys (u,v) and (v,u) and targets (u,v), where (u,v) is in the graph edges list."""
def get_sym_edges_dict(G):
    result = {}
    for u,v in G.edges:
        result[(u,v)] = (u,v)
        result[(v,u)] = (u,v)
    return result

"""Outputs the LCC size of the input graph."""
def LCC(G):
    return len(max(nx.connected_components(G), key=len))


"""Function to build a graph from a connected component of a cut graph. The whole graph must be a simple Graph."""
def build_graph_from_component(whole_graph, component, original_weight_dict:dict, original_length_dict:dict, former_dict:dict):
    nformer_dict = {}
    for former_node in former_dict.keys():
        nformer_dict[former_node] ={"former": former_dict[former_node]}
    edge_list = []
    weight_dict = {}
    length_dict = {}
    for edge in whole_graph.edges:
        if edge[0] in component or edge[1] in component:
            edge_list.append(edge)
            weight_dict[edge] = {"weight" : original_weight_dict[edge]}
            length_dict[edge] = {"length" : original_length_dict[edge]}
    G = nx.Graph(edge_list)
    nx.set_edge_attributes(G, weight_dict)
    nx.set_edge_attributes(G, length_dict)
    nx.set_node_attributes(G, nformer_dict)
    G_result = nx.convert_node_labels_to_integers(G, first_label=0)
    return G_result

"""Plots min, max, mean and median of the input y values as a function of the input x values as bars."""
def plot_stats(x:list, y:list, labels:list, plot_name:str):
    mins = []
    maxs = []
    means = []
    medians = []
    for y_ in y:
        mins.append(min(y_))
        maxs.append(max(y_))
        means.append(sum(y_)/len(y_))
        medians.append(np.median(y_))

    xrange = max(x) - min(x)
    x = np.array(x)
    plt.figure()
    # Min–max vertical line
    plt.vlines(x, mins, maxs, linewidth=1, color = 'black')
    plt.hlines(mins, x - 0.01*xrange, x + 0.01*xrange, linewidth=1, color='black')
    plt.hlines(maxs, x - 0.01*xrange, x + 0.01*xrange, linewidth=1, color='black')
    # Mean (solid horizontal tick)
    plt.hlines(means, x - 0.01*xrange, x + 0.01*xrange, linewidth=1, color='black', label="mean")
    # Median (dashed horizontal tick)
    plt.hlines(medians, x - 0.02*xrange, x + 0.02*xrange, linewidth=1, linestyles="dashed", color='black', label="median")
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    # if "cost" in labels:
    #     yvisible = 0
    #     for y in maxs:
    #         if y > yvisible and y < 10000:
    #             yvisible = y
    #     plt.ylim(-10, yvisible+10)
    plt.tight_layout()
    plt.savefig(path(plot_name), dpi=300)
    plt.close()


"""Computes the inverse cumulative distribution of a distribution and returns it with the x values in the log space."""
def compute_icdf(distribution, x_size:int, logscale=True):
    distribution_array = np.sort(np.array(distribution))
    if logscale:
        xmin = np.min(np.where(distribution_array > 0, distribution_array, np.inf))
        xvalues = np.logspace(np.log10(xmin), 0, x_size)
    else:
        xvalues = np.linspace(distribution_array[0], distribution_array[-1], x_size)

    yvalues = np.zeros(x_size)
    for i in range(x_size):
        yvalues[i] = np.sum(np.where(distribution_array < xvalues[i], 1, 0))
    yvalues /= len(distribution_array)
    
    return xvalues, yvalues

"""Outputs the partition of the input graph according to the input cut. k is needed so we only get k blocks, when there are some unconnected leftovers."""
def cut_to_blocks(G, cut:set, k:int):
    G_ = nx.Graph()
    G_.add_nodes_from(G.nodes(data=False))
    G_.add_edges_from([edge for edge in G.edges(data=False) if edge not in cut])
    return sorted(list(nx.connected_components(G_)), key=len, reverse=True)[:k]

def most_common(lst, score=False):
    """Returns a sorted items list in decreasing occurences of items in argument list of list of items. Returns a dictionnary {obj:nb of occ} instead if score=True."""
    data_temp = collections.Counter(list(itertools.chain.from_iterable(lst))).most_common()
    if score:
        data_dict = {}
        for i in range(len(data_temp)):
            data_dict[data_temp[i][0]] = data_temp[i][1]
        return data_dict
    else:
        data = []
        for i in range(len(data_temp)):
            data.append(data_temp[i][0])
        return data
    
def find_best_cuts(cuts:list, weight_dict:dict, verbose:bool=False):
    """Returns the cuts with the lowest cost from the input cuts list."""
    min = np.inf
    result = []
    for cut in cuts:
        cost = 0
        for edge in cut:
            cost += weight_dict[edge]
        if cost < min:
            min = cost
            result = [cut]
        elif cost == min:
            result.append(cut)
    if verbose:
        print(f"Found {len(result)} cuts of minimum cost {min}.")
    return result

def merge_attack_jsons(main_filename:str, secondary_filenames:list, keys_path:list):
    """Merges missing results in the main json at branch_key level from the secondary ones, then deletes the laters. Always checks for the format to be the same, and uses the
    description/content root formatting."""
    main_json = read_json(path(main_filename))
    for sec_name in secondary_filenames:
        sec_json = read_json(path(sec_name))
        main_json_ = main_json[keys_path[0]]
        sec_json_ = sec_json[keys_path[0]]
        for key in keys_path[1:]:
            main_json_ = main_json_[key]
            sec_json_ = sec_json_[key]
        main_json_.update(main_json_ | sec_json_)
        os.remove(path(sec_name))
    write_json(main_json, path(main_filename))

if __name__ == "__main__":
    pass
    
