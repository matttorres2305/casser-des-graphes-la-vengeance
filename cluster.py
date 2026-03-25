import numpy as np
import numpy.ma as ma
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import os
import json
import time
import sys
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
from numba import jit, vectorize

from utils import *

proj_epsg = 'epsg:2154'


"""Outputs the Chamfer distance between two cuts i.e. list of edges of a graph G."""
def chamfer_distance(cut1:list, cut2:list, x_dict:dict, y_dict:dict, distance_type:str):
    xarray1, xarray2 = np.vectorize(x_dict.get)(np.array(cut1)), np.vectorize(x_dict.get)(np.array(cut2))
    yarray1, yarray2 = np.vectorize(y_dict.get)(np.array(cut1)), np.vectorize(y_dict.get)(np.array(cut2))
    x1 = (xarray1[:,0] + xarray1[:,1]) / 2 # longitude if unprojected
    y1 = (yarray1[:,0] + yarray1[:,1]) / 2 # latitude if unprojeted
    x2 = (xarray2[:,0] + xarray2[:,1]) / 2
    y2 = (yarray2[:,0] + yarray2[:,1]) / 2
    x1, x2 = np.broadcast_arrays(np.expand_dims(x1, 0), np.expand_dims(x2, 1))
    y1, y2 = np.broadcast_arrays(np.expand_dims(y1, 0), np.expand_dims(y2, 1))
    if distance_type == "haversine":
        distances_array = ox.distance.great_circle(y1, x1, y2, x2)
    elif distance_type == "euclidean":
        distances_array = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    else:
        print('Wrong distance_type provided')
        sys.exit()
    return np.sum(np.min(distances_array, axis = 1)) + np.sum(np.min(distances_array, axis = 0))

""""Builds the Chamfer distance matrix from stored kahip cuts."""
def make_chamfer_array(cuts_filename:str, graph_filename:str, result_filename:str,   
                       distance_type:str="haversine"):
    assert distance_type in ["euclidean", "haversine"]
    start = time.time() 
    cut_list = read_file(path(cuts_filename, 'cuts'))
    n = len(cut_list)
    G = nx.MultiGraph(nx.read_gml(path(graph_filename)))
    edges_to_remove = []
    for edge in G.edges:
        if str(edge[1]) == str(len(G)-1):
            edges_to_remove.append(edge)
    G.remove_edges_from(edges_to_remove)
    G.remove_node(str(len(G)-1))
    if distance_type == "euclidean":
        G.graph['crs'] = ox.settings.default_crs
        G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris
    x_dict, y_dict = nx.get_node_attributes(G, 'x'), nx.get_node_attributes(G, 'y')
    chamfer_distance_array = np.ones((n, n))*(-1)
    for i in range(0, n):
        for j in range(i+1, n):
            chamfer_distance_array[i, j] = chamfer_distance(cut_list[i], cut_list[j], x_dict, y_dict, distance_type)
        print(f'Computed {i}/{n} of the Chamfer distance array')
    result_path = path(result_filename)
    with open(result_path, 'wb') as f:
        np.save(f, chamfer_distance_array)
    print(f'Chamfer array done in {time.time() - start} s. Saved at {result_path}.')

def compute_zone_distances(blocks_name:str, graph_name:str, distances_name:str, result_name:str):
    start = time.time()
    blocks = read_file(path(blocks_name, 'cuts'))
    n = len(blocks)
    G = nx.read_gml(path(graph_name, 'graphs'))
    print("Transforming blocks into masks.")
    imasks = np.zeros((n, len(G.nodes)-1))
    for i,block in enumerate(blocks):
            out = block[-1]
            if out == 1: # hardcoded for k=2
                imasks[i] = (1 - np.array(block))[:-1]
            else:
                imasks[i] = np.array(block[:-1])
    distances_array = np.load(path(distances_name))
    print("Computing array")
    topo_array = numba_topo_array(imasks, distances_array, n)
    result_path = path(result_name)
    with open(result_path, 'wb') as f:
        np.save(f, topo_array)
    print(f'Array done in {time.time() - start} s. Saved at {result_path}.')

@jit(nopython=True)
def numba_topo_array(imasks:np.ndarray, distances_array:np.ndarray, n:int):
    topo_array = np.ones((n,n)) * (-1)
    for i in range(1,n):
        for j in range(i+1, n):
            mini_dist_array = np.zeros((int(np.sum(imasks[i])), int(np.sum(imasks[j]))))
            mini = 0
            for ip,ival in enumerate(imasks[i]):
                    if int(ival) == 1:
                        minj = 0
                        for jp,jval in enumerate(imasks[j]):
                                if int(jval) == 1:
                                    mini_dist_array[mini, minj] = distances_array[ip,jp]
                                    minj += 1
                        mini += 1
            dist1 = np.zeros(int(np.sum(imasks[i])))
            for k in range(int(np.sum(imasks[i]))):
                dist1[k] = np.min(mini_dist_array[k,:])
            dist2 = np.zeros(int(np.sum(imasks[j])))
            for k in range(int(np.sum(imasks[j]))):
                dist2[k] = np.min(mini_dist_array[:,k])
            topo_array[i, j] = np.sum(dist1) + np.sum(dist2)
    return topo_array


"""Plots the distance distribution from the chamfer distance matrix."""
def plot_array_stat(array_filename:str, plot_name:str, cumsum=True, save_data_filename:str = "", zoom = None, l=None):
    with open(path(array_filename), 'rb') as f:
        chamfer_distance_array = np.load(f)
    n = chamfer_distance_array.shape[0]
    positive_values = []
    for i in range(n):
        for j in range(i+1, n):
            if chamfer_distance_array[i,j] >= 0:
                positive_values.append(chamfer_distance_array[i,j])
    if save_data_filename:
        write_file(positive_values, path(save_data_filename))

    if cumsum:
        xvalues, yvalues = compute_icdf(positive_values, 10000, logscale=False)

        plt.figure()
        plt.plot(xvalues, yvalues, label = 'distribution')
        if l:
            plt.vlines(l, min(yvalues), max(yvalues), linestyles='dotted', label=rf"$l={l}$")
            plt.legend(loc='lower right')
        plt.xlabel('distances between cuts')
        plt.ylabel('cumulative distribution of distances')

        plt.minorticks_on()  # Active les ticks mineurs
        
    else:
        fig, (ax1) = plt.subplots(1)
        
        sns.histplot(positive_values, kde=False, ax=ax1)
        ax1.set_xlabel('distances between cuts')
        ax1.set_ylabel('distribution of distances')
        plt.grid(axis='x')

        plt.minorticks_on()  # Active les ticks mineurs
        plt.xticks(np.arange(10) * 50000)  # tous les 1 unité sur x
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # Incline les labels à 45°

    if zoom:
        plt.xlim(0, max(xvalues)/zoom)
    plt.tight_layout()

    result_path = path(plot_name)
    plt.savefig(result_path, dpi=300)
    print(f'Array distribution saved at {result_path}.')

def get_zone_distances(blocks_name:str, graph_name:str, distances_name:str, result_name:str, plot_name:str=""):
    """Calcule le tableau des distances de zone entre chaque paire de coupes-(2,epsilon) associé à l'ensemble des blocs fournis en entrée et en affiche
    facultativement la distribution cumulative. Utilise un tableau de distances entre noeuds précalculé par 'model_graph'.
    'blocks_name' et 'graph_name' correspondent au nom respectif des fichiers sources des blocs correspondant aux coupes et du graphe;
    'distances_name' à celui du fichier contenant le tableau des distances entre noeuds;
    'results_name' à celui du fichier du tableau résultant;
    'plot_name' est facultatif et correspond au nom du plot résultant.
    Cette fonction peut nécessiter un temps de calcul élevé."""
    assert result_name.split(sep=".")[-1] == "npy"
    compute_zone_distances(blocks_name, graph_name, distances_name, result_name)
    if plot_name:
        plot_array_stat(array_filename=result_name, plot_name=plot_name)


if __name__ == "__main__":
    pass
    city = "paris"
    extension = "_frac0.25"
    n = 1000
    k = 2
    imb = 0.02
    distance = "topological"
    
    # Partsim array
    compute_zone_distances(blocks_filename=f'blocks{n}_k{k}_imb{imb}{extension}',
                       graph_filename=f'graph_{city}{extension}',
                       distances_filename=f"distances_paris_{distance}.npy",
                       result_filename=f"array_{distance}_cuts{n}_k{k}_imb{imb}{extension}.npy")
    plot_array_stat(array_filename=f"array_{distance}_cuts{n}_k{k}_imb{imb}{extension}.npy",
                    plot_name=f"distrib_{distance}_cuts{n}_k{k}_imb{imb}{extension}.png",
                    )

    # # Birch clustering
    # G = nx.read_gml(path('graph_clean_shanghai'))
    # imb = 0.21
    # cut_list = read_file(path(f'cuts1000_k2_imb{imb}_shanghai'))
    # md = 1000000
    # with open(path(f'chamfer_array_imb{imb}_shanghai'), 'rb') as f:
    #     chamfer_array = np.load(f)
    # distances_array = chamfer_array + chamfer_array.transpose() + np.diag(np.ones(chamfer_array.shape[0])) + 1
    # birch_clustering_pipeline(max_diameter=md,
    #                           result_name=f'clusters_birch_shanghai_md{md}_imb{imb}',
    #                           cut_list=cut_list,
    #                           graph=G,
    #                           distances_array=chamfer_array)

    # # Plot of clusters on the graph
    # md = 25000
    # com03 = read_file(path('clusters_l25000_imb0.1', 'clusters'))[:4]
    # cuts03 = read_file(path('cuts1000_k2_imb0.1_mode2_clean', 'cuts'))
    # G = nx.MultiGraph(nx.read_gml(path("graph_paris_clean")))
    # G.graph['crs'] = ox.settings.default_crs
    # G = ox.project_graph(G, to_crs='epsg:2154') ## pour le mettre dans le même référentiel que les données de Paris
    # edge_keys = list(G.edges)
    # edgecolor_dict = dict.fromkeys(edge_keys, 'gray')
    # large_dict = dict.fromkeys(edge_keys, 0.3)
    # color_dict = {0:"black", 1:"magenta", 2:"red", 3:"orange"}
    # custom_lines = []
    # legend = []
    # custom_lines.append(Line2D([0], [0], color=color_dict[0], lw=4))
    # legend.append(r"$i=1$")
    # custom_lines.append(Line2D([0], [0], color=color_dict[1], lw=4))
    # legend.append(r"$i=2$")
    # custom_lines.append(Line2D([0], [0], color=color_dict[2], lw=4))
    # legend.append(r"$i=3$")
    # custom_lines.append(Line2D([0], [0], color=color_dict[3], lw=4))
    # legend.append(r"$i=4$")
    # com_list = [com03]
    # cut_list = [cuts03]
    # for i in range(1):
    #     for com_id in range(len(com_list[i])):
    #         for cut_id in com_list[i][com_id]:
    #             for edge in cut_list[i][int(cut_id)]:
    #                 edge = (edge[0], edge[1], 0)
    #                 edgecolor_dict[edge] = color_dict[com_id+i*2]
    #                 large_dict[edge] = 2
    # plt.figure()
    # ox.plot.plot_graph(G, edge_color=list(edgecolor_dict.values()), node_size=0.01, edge_linewidth=list(large_dict.values()), bgcolor = 'white')
    # plt.legend(custom_lines, legend)
    # plt.savefig(path("paris_clusters.png"), dpi=300)
    # plt.close()
    

    # # Plot of a md value resulting clusters distances distributions
    # for md in [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]:
    #     plot_clusters_distribution(clusters_filename=f'clusters_birch_C_md{md}_clean0.03',
    #                             cutlist_filename='cuts1000_k2_imb0.03_mode2_clean',
    #                             plot_name=f'distribution_clusters_birch_C_md{md}_clean0.03.png',
    #                             array_filename='chamfer_array_C_clean0.03',
    #                             save_data_filename=f'distribution_nocumsum_data_clusters_birch_C_md{md}_clean0.03',
    #                             cumsum=False,)

    # # Plot all clusters distances distribution compared to the individual cuts one
    # template_start = 'distribution_data_clusters_birch_C_md'
    # template_end = '_clean0.03'
    # filename_list = ["distribution_data_chamfer_clean0.03"]
    # for md in [5000, 10000, 15000, 20000, 25000, 30000, 35000]:
    #     filename_list.append(template_start+str(md)+template_end)
    # plot_clusters_distribution(clusters_filename=None,
    #                            cutlist_filename=None,
    #                            array_filename=None,
    #                            plot_name=f'distribution_nocumsum_clusters_birch_C_clean0.03.png',
    #                            cumsum=False,
    #                            data_filename_list=filename_list)

    # Plot of the average cluster diameter
    # md = [5000, 10000, 15000, 20000, 25000, 30000, 35000]
    # plot_clusters_diameters(clusters_filename=f'clusters_birch_C_md25000_clean0.03',
    #                         array_filename='chamfer_array_C_clean0.03',
    #                     plot_name="diameterssize_clusters_C.png",
    #                     plot_by_size=True,
    #                     md_list=md)

    # plot_clusters_costs(clusters_filename=f'clusters_birch_shanghai_md{md}_imb0.1',
    #                     cuts_filename='cuts1000_k2_imb0.1_shanghai',
    #                     graph_filename="graph_clean_shanghai",
    #                     plot_name=f"costssize_clusters_shanghai_md{md}.png",
    #                     plot_by_size=True
    #                     )

    # plot_clusters_diameters(clusters_filename=f'clusters_birch_shanghai_md{md}_imb0.1',
    #                         array_filename='chamfer_array_imb0.1_shanghai',
    #                         l_value=md,
    #                     plot_name=f"diameterssize_clusters_shanghai_md{md}.png",
    #                     plot_by_size=True)

    # size_array = np.zeros(n)
    # for i in range(n):
    #     size_array[i] = len(clusters_list[i])

    # values = size_array.flatten()
    
    # fig, (ax1) = plt.subplots(1)
    
    # sns.histplot(values, kde=False, ax=ax1)
    # ax1.set_xlabel('Cluster size')
    # ax1.set_ylabel('Frequency')

    # plt.grid(axis='x')
    # plt.minorticks_on()
    # plt.xticks(np.arange(2) * 10)  # tous les 1 unité sur x
    # plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # Incline les labels à 45°
    
    # plt.xlim(140, 220)

    # plt.figure()
    # plt.scatter(np.arange(n), size_array, marker = '+')
    # plt.xlabel('cluster')
    # plt.ylabel('size')

    # plt.tight_layout()
    # plt.savefig(path(f'size(cluster)_clusters_birch_md{md}_fclean0.03.png'), dpi=300)
    # plt.close()
    
    # # plot_clusters(filepath, graph_name, f'clean0.03_birch_clusters_{max_diameter}_joined.png', cut_list, clusters_list, one_com=False)
    
    # chamfer_array_name = 'chamfer_array_fclean0.03'
    # with open(os.path.join(filepath, chamfer_array_name), 'rb') as f:
    #     chamfer_array = np.load(f)
    # clusters_silhouette_coef(clusters_list, chamfer_array, f'clusters_birch_silhouette_md{md}_fclean0.03.png')

    
    # best_cuts_list = find_best_cuts(filepath, graph_name, cut_list, 147, plot_name = "clean0.03_Paris_best_cuts.png")
    # print(len(best_cuts_list))
    # write_file(best_cuts_list, os.path.join(filepath, 'cleancuts_1000_k2_imb0.03_mode2_bestcuts147'))

    # clusters_silhouette_coef(communities_list, G_chamfer, chamfer_array, f'clean0.03_clusters_{threshold}_silhouette.png')
    
    
    
    # with open(os.path.join(filepath, chamfer_array_name), 'rb') as f:
    #     chamfer_array = np.load(f)

    # threshold = 20000

    # G_chamfer = nx.read_gml(os.path.join(filepath,chamfer_graph_name))

    # communities_list = nx.community.louvain_communities(G_chamfer, seed = 0)
    # communities_list.sort(key=len, reverse=True)
    # l = len(communities_list)
    # print(f"{l} communities found")

    # # cluster_id = 0
    # # communities_list = [communities_list[cluster_id]]

    # # # node_to_community = {}
    # # # for i in G_chamfer.nodes:
    # # #     for j in range(l):
    # # #         if i in communities_list[j]:
    # # #             node_to_community[i] = j

    # cut_list = read_file(os.path.join(filepath, cut_name))
    # # best_cuts_list = find_best_cuts(filepath, graph_name, cut_list, 147, plot_name = "clean0.03_Paris_best_cuts.png")
    # # print(len(best_cuts_list))
    # # # write_file(best_cuts_list, os.path.join(filepath, 'cleancuts_1000_k2_imb0.03_mode2_bestcuts147'))