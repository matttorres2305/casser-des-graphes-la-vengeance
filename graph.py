import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import time
from scipy import stats
import seaborn as sns

from utils import *

# Some magic to make kahip work
import sys
sys.path.append('/home/torres/.vscode-server/data/User/workspaceStorage/KaHIP/deploy')

import kahip

place = 'Paris, Paris, France'
epsg_global = 'epsg:4326'
epsg_paris = 'epsg:2154'
epsg = epsg_paris


"""Uses OSMnx to output a raw urban MultiDiGraph from a place (Paris)."""
def import_graph(place:str, graph_name:str, buffer:int=350):
    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = ox.utils_geo.buffer_geometry(gdf.iloc[0]["geometry"], buffer)
    G = ox.graph.graph_from_polygon(polygon, network_type="drive", simplify=False,retain_all=True,truncate_by_edge=False)
    G_place = ox.project_graph(G, to_crs=epsg) ## pour le mettre dans le même référentiel que les données de Paris
    G2 = ox.consolidate_intersections(G_place, rebuild_graph=True, tolerance=4, dead_ends=True)
    G_out = ox.project_graph(G2, to_crs=epsg_global)
    toremove_list = []
    for node in G_out.nodes:
        if G_out.degree(node) == 0:
            toremove_list.append(node)
    G_out.remove_nodes_from(toremove_list)
    G_result = nx.convert_node_labels_to_integers(G_out, first_label=0)
    ox.save_graphml(G_result, filepath=path(graph_name, 'graphs'))

"""Takes the saved raw MultiDiGraph as input, and process it to output the weighted simple graph."""
def weight_and_simplify_graph(raw_graph_name:str, simple_graph_name:str):
    G = ox.load_graphml(path(raw_graph_name, 'graphs'))
    highway_dict = nx.get_edge_attributes(G, "highway", default="None")
    lanes_dict = nx.get_edge_attributes(G, "lanes", default=-1)
    length_dict = nx.get_edge_attributes(G, "length", default=-1)
    weight_dict = {}
    for edge in G.edges:
        weight = 2
        if edge in highway_dict.keys():
            highway = highway_dict[edge]
            if highway == "primary" or highway == "secondary":
                weight = 3
        if edge in lanes_dict.keys():
            lanes = lanes_dict[edge]
            if int(lanes) >= 0:
                weight = lanes
        weight_dict[edge] = {"weight" : weight}
    # Creating the new graph before the projection to avoid conflicts
    nx.set_edge_attributes(G, weight_dict)
    new_weight_dict = {}
    new_length_dict = {}
    edge_list = []
    for u,v,z in G.edges:
        if (u,v) not in edge_list and (v,u) not in edge_list:
            edge_list.append((u,v))
            weight = 0
            length_list = []
            if (u,v,2) in G.edges or (v,u,2) in G.edges:
                km = 2
            elif (u,v,1) in G.edges or (v,u,1) in G.edges:
                km = 1
            else:
                km = 0
            for k in range(km + 1):
                if (u,v,k) in G.edges:
                    weight += int(weight_dict[(u,v,k)]['weight'])
                    length_list.append(float(length_dict[u,v,k]))
                if (v,u,k) in G.edges:
                    weight += int(weight_dict[(v,u,k)]['weight'])
                    length_list.append(float(length_dict[v,u,k]))
            new_weight_dict[(u,v)] = {"weight" : weight}
            new_length_dict[(u,v)] = {"length" : min(length_list)}
    new_graph = nx.Graph()
    new_graph.add_nodes_from(G.nodes(data=True))
    new_graph.add_edges_from(edge_list)
    nx.set_edge_attributes(new_graph, new_weight_dict)
    nx.set_edge_attributes(new_graph, new_length_dict)
    nx.write_gml(new_graph, path(simple_graph_name, 'graphs'))

"""Takes a saved simple graph as input to delete every path of degree 2 and returns the clean graph as output."""
def clean_graph(input_graph_name:str, output_graph_name:str, output_ncleantoclean_dict_name:str, output_cleantonclean_dict_name:str, verbose:bool = False):
    G = nx.read_gml(path(input_graph_name, 'graphs'))
    weight_dict = nx.get_edge_attributes(G, 'weight')
    length_dict = nx.get_edge_attributes(G, 'length')
    neighbors_dict = find_neighbors(G, weight_dict)
    sym_dict = get_sym_edges_dict(G)

    notclean_to_clean_dict = {}
    clean_to_notclean_dict = defaultdict(list)
    weightlength_dict = defaultdict(lambda : {'weight' : 0, 'length' : +np.inf}) # key = new edges, target = {'weight' : weight, 'length' = length}
    visited_edges = []
    for node in G.nodes:
        if verbose and (len(G.nodes)//int(node)) % 10 == 0:
            print((len(G.nodes)//int(node)) // 10)
        if G.degree[node] != 2:
            for tuple in neighbors_dict[node]:
                if sym_dict[(node, tuple[0])] not in visited_edges:
                    start_node = node
                    explored_node = tuple[0]
                    explored_edges = [sym_dict[(node, tuple[0])]]
                    wmin = tuple[1]
                    length = length_dict[sym_dict[(node, tuple[0])]]
             
                    while G.degree[explored_node] == 2:
                        iter_gen = iter(neighbors_dict[explored_node])
                        explored_tuple = next(iter_gen)
                        if sym_dict[(explored_tuple[0], explored_node)] in explored_edges:
                            explored_tuple = next(iter_gen)
                        length += length_dict[sym_dict[(explored_node, explored_tuple[0])]]
                        if explored_tuple[1] < wmin:
                            wmin = explored_tuple[1]
                        explored_edges.append(sym_dict[(explored_node, explored_tuple[0])])
                        explored_node = explored_tuple[0]

                    end_node = explored_node
                    small_c_to_nc_list = []
                    if (end_node, start_node) in clean_to_notclean_dict.keys():
                        end_node_copy = end_node
                        end_node = start_node
                        start_node = end_node_copy
                    if (start_node, end_node) in clean_to_notclean_dict.keys() and (start_node, end_node) in G.edges:
                        wmin += weight_dict[sym_dict[(start_node,end_node)]]
                        weightlength_dict[(start_node, end_node)]['length'] = min(weightlength_dict[(start_node, end_node)]['length'], length_dict[sym_dict[(start_node,end_node)]])
                    if (start_node, end_node) in clean_to_notclean_dict.keys() and (start_node, end_node) in G.edges and weightlength_dict[(start_node, end_node)]['weight'] < 10000:
                        small_c_to_nc_list.append((start_node, end_node))
                    for edge in explored_edges:
                        visited_edges.append(edge)
                        notclean_to_clean_dict[edge] = (start_node, end_node)
                        if weight_dict[edge] == wmin and wmin < 10000:
                            small_c_to_nc_list.append(edge)
                    clean_to_notclean_dict[(start_node, end_node)] += small_c_to_nc_list
                    weightlength_dict[(start_node, end_node)]['weight'] += wmin
                    weightlength_dict[(start_node, end_node)]['length'] = min(weightlength_dict[(start_node, end_node)]['length'], length)
    write_file(clean_to_notclean_dict, path(output_cleantonclean_dict_name, 'graphs'))
    write_file(notclean_to_clean_dict, path(output_ncleantoclean_dict_name, 'graphs'))
    new_graph = nx.Graph()
    new_graph.add_nodes_from(G.nodes(data=True))
    for edge in weightlength_dict.keys():
        new_graph.add_edge(edge[0], edge[1], weight = weightlength_dict[edge]['weight'], length = weightlength_dict[edge]['length'])
    toremove_list = []
    for node in new_graph.nodes:
        if new_graph.degree(node) == 0:
            toremove_list.append(node)
    new_graph.remove_nodes_from(toremove_list)
    toremove_list = []
    for u,v in new_graph.edges:
        if u == v:
            toremove_list.append((u,v))
    new_graph.remove_edges_from(toremove_list)
    edge_label_dict = {}
    for edge in new_graph.edges:
        edge_label_dict[edge] = {'former_name':edge}
    nx.set_edge_attributes(new_graph, edge_label_dict)
    LCC = sorted(list(nx.connected_components(new_graph)), reverse=True)[0]
    tr = []
    for node in new_graph.nodes:
        if node not in LCC: tr.append(node)
    new_graph.remove_nodes_from(tr)
    G_result = nx.convert_node_labels_to_integers(new_graph, first_label=0)
    relabelled_graph = nx.convert_node_labels_to_integers(G_result, label_attribute='former_name')
    clean_path = path(output_graph_name, 'graphs')
    nx.write_gml(relabelled_graph, clean_path)

"""Uses the unprojected polygon from the place to detect the input graph nodes that are outside the limits of the city."""
def detect_boundaries(input_graph:str, place:str, output_graph:str, plot:bool=False):
    gdf = ox.geocoder.geocode_to_gdf(place)
    polygon = gdf.iloc[0]["geometry"]
    G = nx.MultiGraph(nx.read_gml(path(input_graph, 'graphs')))
    G.graph['crs'] = epsg_global
    nodes, _ = ox.graph_to_gdfs(G)
    nodes["inside"] = nodes.within(polygon)
    nx.set_node_attributes(G, nodes["inside"].to_dict(), "inside")
    G_result = nx.Graph(G)
    G_result.add_nodes_from([(len(G_result),{'ghost':1})])
    edges_to_add = []
    for node in list(G_result.nodes(data=True))[:-1]:
        if not node[1]["inside"]:
            edges_to_add.append((node[0], len(G_result)-1))
    G_result.add_edges_from(edges_to_add, weight=10000, ghost=1)
    nx.write_gml(G_result, path(output_graph, 'graphs'))
    if plot:
        plot_graph(graph_filename=output_graph, plot_name=f"{output_graph}_boundaries.png", highlight_boundaries=True)

"""Computes the distances matrix between every nodes of the input bound graph."""
def compute_distances(graph_name:str, array_name:str, distance_type:str):
    assert distance_type in ['euclidean', 'topological'] # Topological not implemented
    print("Computing distances.")
    G = nx.read_gml(path(graph_name, "graphs"))
    G.remove_node(str(len(G)-1))
    distance_array = np.zeros((len(G.nodes), len(G.nodes)))
    if distance_type == 'euclidean':
        G = nx.MultiGraph(G)
        G.graph["crs"] = epsg_global
        G = ox.project_graph(G, to_crs=epsg_global)
        for node1 in list(G.nodes(data=True)):
            x1 = node1[1]['x']
            y1 = node1[1]['y']
            for node2 in list(G.nodes(data=True)):
                x2 = node2[1]['x']
                y2 = node2[1]['y']
                distance_array[int(node1[0]), int(node2[0])] = ox.distance.euclidean(y1, x1, y2, x2)
    elif distance_type == 'topological':
        lengths = nx.all_pairs_dijkstra_path_length(G, weight = "length")
        for source, targets in lengths:
            for target, distance in targets.items():
                distance_array[int(source), int(target)] = distance
    np.save(path(array_name), distance_array)

"""Weights the nodes of the input graph with the given parameters and a normalization."""
def weight_objective(input_graph_name:str, output_graph_name:str, distances_array_name:str, objective_node:int, objective_buffer:float, alpha:float):
    G = nx.read_gml(path(input_graph_name, 'graphs'))
    weight_dict = {}
    distances_array = np.load(path(distances_array_name))
    n = len(G.nodes)
    if objective_node == None:
        for node in G.nodes:
            weight_dict[node] = {"weight":1}
    else:
        objective = objective_node
        nodes_list = []
        nb = 0
        for node in list(G.nodes(data=True))[:-1]:
            if node[1]["inside"] == False:
                nb += 1
            if distances_array[objective, int(node[0])] < objective_buffer:
                if node[1]["inside"] == False:
                    print('Area too close to outside. Please change the value of the objective_node parameter.')
                    sys.exit()
                nodes_list.append(node[0])
            else:
                weight_dict[node[0]] = {"weight":1}
            if int(node[0]) == len(G)-2:
                no = len(nodes_list)
                ns = n - no - nb - 1
        we = 3 * ns - nb - no
        wo = (nb + we + (1 - 2 * alpha) * ns) / no
        for node in nodes_list:
            weight_dict[node] = {"weight":int(wo)}
        weight_dict[str(len(G)-1)] = {"weight":int(we)}
        for edge in G.edges(data=True):
            if weight_dict[edge[0]]['weight'] > 1 and weight_dict[edge[1]]['weight'] > 1:
                edge[2]["weight"] = 10000
    nx.set_node_attributes(G, weight_dict)
    nx.write_gml(G, path(output_graph_name, 'graphs'))

# """Takes a clean graph as input and output the kahip adjacency list file."""
def parse_graph_to_kahip(graph_name, result_name, weight_objective:bool=True):
    print("Parsing graph into kahip.")
    start = time.time()

    G = nx.read_gml(path(graph_name, 'graphs'))
    weight_dict = nx.get_edge_attributes(G, "weight")

    n = str(len(G.nodes))
    m = str(len(G.edges))
    if weight_objective:
        f = "11"
    else:
        f = "1"
    
    nodes_neighbors_edges = {}
    for edge in G.edges:
        if edge[0] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[0]] = set()
        nodes_neighbors_edges[edge[0]].add((edge[1], weight_dict[edge]))
        if edge[1] not in nodes_neighbors_edges.keys():
            nodes_neighbors_edges[edge[1]] = set()
        nodes_neighbors_edges[edge[1]].add((edge[0], weight_dict[edge]))
    # print(nodes_neighbors_edges)

    result_path = path(result_name, 'graphs')
    with open(result_path, "w") as file:
        file.write(n+" "+m+" "+f+ "\n")
        for node in range(len(nodes_neighbors_edges.keys())):
            line = ""
            if weight_objective:
                line+=str(list(G.nodes(data=True))[node][1]["weight"])+" "
            for elem in nodes_neighbors_edges[str(node)]:
                line+=str(elem[0])+" "+str(elem[1])+" "
            file.write(line + "\n")

    print(f"Graph parsed in {time.time() - start} s. Saved at {result_path}.")

"""Plots the input graph. Highlighted nodes are in red, and boundaries nodes in blue."""
def plot_graph(graph_filename:str, plot_name:str, proj_epsg:str="", graph_type:str = "gml", highlight_nodes:list=[], highlight_boundaries:bool=False):
    if graph_type == "gml":
        G = nx.MultiGraph(nx.read_gml(path(graph_filename, 'graphs')))
    elif graph_type == "graphml":
        G = nx.MultiGraph(ox.load_graphml(path(graph_filename, 'graphs')))
    else:
        print(f"{graph_type} is wrong. Should be 'gml' or 'graphml'.")
        sys.exit()

    if proj_epsg and graph_type=='gml':
        G.graph["crs"] = proj_epsg
        G = ox.project_graph(G, to_crs=proj_epsg)

    if highlight_nodes or highlight_boundaries:
        edge_width = 0
    else:
        edge_width = 0.3
    color_list = ['gray']*len(G.nodes)
    large_list = [0.5]*len(G.nodes)
    for node in G.nodes(data=True):
        if node[0] in highlight_nodes:
            color_list[int(node[0])] = 'red'
            large_list[int(node[0])] = 8
        if highlight_boundaries:
            try:
                if not node[1]["inside"]:
                    color_list[int(node[0])] = 'blue'
                    large_list[int(node[0])] = 8
            except:
                pass
    plt.figure()
    ox.plot.plot_graph(G, node_size=large_list, node_color=color_list, edge_linewidth=edge_width, bgcolor = 'white')
    plt.savefig(path(plot_name), dpi=300)
    plt.close()

def model_graph(result_name:str, place:str, buffer:int, objective_node:int=None, objective_buffer:float=0.005, alpha:float=0.):
    """Pipeline complète comprenant l'importation des données d'OSM, la pondération des arêtes du graphe, leur simplicafication, la modélisation du réseau extérieur,
    et la pondération facultative des noeuds et arêtes nécessaires à la modélisation d'une zone-objectif. Calcule également deux tableaux contenant toutes les distances,
    euclidiennes et topologiques, entre chaque paire de noeuds, utiles notamment pour le calcul des distances de zone. L'opération a lieu dans un dossier 'graphs' créé
    automatiquement s'il n'existe pas déjà.
    'place' est hardcodé par OSM et doit rester identique à la valeur par défaut pour le graphe de Paris, et 'buffer' correspond à la taille de la bande extérieure en mètres ;
    'objective_node' est facultatif et correspond à l'ID du noeud central de la zone-objectif, avec par défaut None pour obtenir le graphe sans zone-objectif ;
    'objective_buffer' et 'alpha' sont facultatifs si 'objective_node'=None, et correspondent respectivement au rayon de la zone-objectif en unité 'latitude-longitude', et à
    la proportion de noeuds internes à la ville de Paris et externes à la zone-objectif qui doivent être séparés du réseau extérieur par une coupe-(2,0.0) (voir rapport).
    Si la combinaison de 'objective_node' et 'objective_buffer' sont incompatibles avec une zone-objectif disjointe de la bande extérieure, la fonction renvoie
    une erreur."""
    if not os.path.exists(path('graphs')):
        os.makedirs(path('graphs'))
    import_graph(place=place, graph_name=f"graph_raw", buffer=buffer)
    weight_and_simplify_graph(raw_graph_name=f"graph_raw", simple_graph_name=f"graph_simple")
    clean_graph(input_graph_name=f"graph_simple", output_graph_name=f"graph_clean",
                output_cleantonclean_dict_name=f"edges_clean_to_simple",
                output_ncleantoclean_dict_name=f"edges_simple_to_clean", verbose=False)
    detect_boundaries(input_graph=f"graph_clean", place=place, output_graph=f"graph_bound", plot=False)
    compute_distances(graph_name=f"graph_bound", array_name=f"distances_euclidean_{result_name}.npy", distance_type="euclidean")
    compute_distances(graph_name=f"graph_bound", array_name=f"distances_topological_{result_name}.npy", distance_type="topological")
    weight_objective(input_graph_name=f"graph_bound", output_graph_name=result_name, distances_array_name=f"distances_euclidean_{result_name}.npy", objective_node=objective_node, objective_buffer=objective_buffer, alpha=alpha)
    parse_graph_to_kahip(graph_name=result_name, result_name=f"kahip_{result_name}", weight_objective=True)
    G = nx.read_gml(path(result_name, "graphs"))
    nodes = []
    for node in G.nodes(data=True):
        if int(node[1]["weight"]) > 1:
            nodes.append(node[0])
    plot_graph(graph_filename=result_name, plot_name=f"objective_{result_name}.png",
               highlight_nodes=nodes)
    
# """From the input dictionnary of edges frequency, outputs and saves a new version of the input graph with infinite weights on the edges above a given frequency."""
# def robust_weighting(graph_name:str, freq_dict_name:str, result_name:str, threshold:float):
#     G = nx.read_gml(path(graph_name))
#     freq_dict = read_json(path(freq_dict_name))["content"]
#     for edge in G.edges:
#         if freq_dict[f"('{edge[0]}', '{edge[1]}')"] > threshold:
#             G[edge[0]][edge[1]]['weight'] = 10000
#     nx.write_gml(G, path(result_name))

if __name__ == "__main__":
    pass
    # # Graph import, weighting and processing
    # model_graph(place=place, buffer=350, weight_objective_type="default")

    # Define objective
    # model_graph(place=place, buffer=350, weight_objective_type="area", frac=0.75, alt_suffix="frac0.75")
    # G = nx.read_gml(path("graph_paris_area0.1", "graphs"))
    # nodes = []
    # for node_ in G.nodes(data=True):
    #     if int(node_[1]["weight"]) > 1:
    #         nodes.append(node_[0])
    # plot_graph(graph_filename="graph_paris_area0.1", plot_name="graph_paris_area0.1.png",
    #            highlight_nodes=nodes)