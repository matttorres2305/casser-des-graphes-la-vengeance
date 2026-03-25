### Si ce fichier est utilisé en dehors de son emplacement original dans samos, il faut modifier la variable "filepath" dans le fichier "utils.py" pour y rentrer le chemin
### absolu du dossier dans lequel tous les fichiers sont situés.
### Les fichiers représentant des graphes sont créés et manipulés dans un dossier "graphs", idem pour les coupes et blocs associés dans un dossier "cuts".
### Les valeurs par défauts sont celles utilisées dans le rapport. 

from graph import *
from cut import *
from cluster import *

if __name__ == "__main__":
    ### graph.py
    graph_name = "paris"
    place = 'Paris, Paris, France'

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
    model_graph(result_name=graph_name, place='Paris, Paris, France', buffer=350, objective_node=100, objective_buffer=0.005, alpha=0.)

    ### cut.py
    n = 1000
    k = 2
    epsilon = 0.1

    """Calcule un ensemble de n coupes-(k,epsilon) à partir d'une représentation "kahip" d'un graphe avec kaffpa, puis les stocke dans un fichier dans un dossier 'cuts' créé
    automatiquement s'il n'existe pas déjà.
    'graph_name' et 'kahip_graph' dénotent le nom des fichiers sources générés par 'model_graph', et 'result_name' celui du fichier final contenant les coupes;
    'n', 'k' et 'epsilon' aux paramètres éponymes des n coupes-(k,epsilon);
    'blocks_name' est facultatif et permet de stocker un fichier contenant les partitions de noeuds en plus des ensembles d'arêtes, utiles pour les calculs de distances
    entre coupes par exemple."""
    make_cuts(graph_name=graph_name, kahip_graph=f"kahip_{graph_name}", result_name=f"cuts{n}_k{k}_imb{epsilon}",
                n=n, k=k, epsilon=epsilon,
                blocks_name=f"blocks{n}_k{k}_imb{epsilon}")
    
    """Calcule la fréquence d'apparition des arêtes d'un graphe dans un ensemble de coupes de ce graphe, et les stocke dans un .json, puis les affiche facultativement sur
    le graphe.
    'cuts_name' et 'graph_name' dénotent respectivement le nom des fichiers sources des coupes et du graphe, et 'json_name' celui du .json résultant;
    'plot_name' est facultatif et correspond au nom du plot résultant."""
    get_freq(cuts_name=f"cuts{n}_k{k}_imb{epsilon}", graph_name=graph_name,
             json_name=f"edgefreq_cuts{n}_k{k}_imb{epsilon}.json", plot_name=f"edgefreq_cuts{n}_k{k}_imb{epsilon}.png")

    """Renvoie la liste des coûts des coupes du graphe fournis en entrée.
    'cuts' et 'graph' peuvent tous les deux contenir le nom du fichier correspondant, ou alors l'objet lui-même : 'cuts' doit alors être une liste de liste (=coupes)
    de tuples (=arêtes) comme stockée par 'make_cuts', et 'graph' un Graph networkx comme stocké par 'model_graph'."""
    costs = get_cost(cuts=f"cuts{n}_k{k}_imb{epsilon}", graph=graph_name)
    print(f"Le coût moyen de l'ensemble de {n} coupes-({k},{epsilon}) du graphe {graph_name} est {sum(costs)/len(costs)}.")

    """Renvoie la liste des proportions des noeuds du graphe contenus dans la partie isolée des coupes correspondantes aux blocs fournis en entrée."""
    prop_isolated = get_isolated(blocks_name=f"blocks{n}_k{k}_imb{epsilon}", graph_name=graph_name)
    print(f"La proportion moyenne de noeuds isolés par l'ensemble de {n} coupes-({k},{epsilon}) du graphe {graph_name} est {sum(prop_isolated)/len(prop_isolated)}.")

    ### cluster.py
    distance_type = "topological" # correspond au type de distance utilisée entre les noeuds, peut être choisie comme "euclidean" ou "topological"

    """Calcule le tableau des distances de zone entre chaque paire de coupes-(2,epsilon) associé à l'ensemble des blocs fournis en entrée et en affiche
    facultativement la distribution cumulative. Utilise un tableau de distances entre noeuds précalculé par 'model_graph'.
    'blocks_name' et 'graph_name' correspondent au nom respectif des fichiers sources des blocs correspondant aux coupes et du graphe;
    'distances_name' à celui du fichier contenant le tableau des distances entre noeuds;
    'results_name' à celui du fichier du tableau résultant;
    'plot_name' est facultatif et correspond au nom du plot résultant.
    Cette fonction peut nécessiter un temps de calcul élevé."""
    get_zone_distances(blocks_name=f"blocks{n}_k{k}_imb{epsilon}", graph_name=graph_name,
                       distances_name=f"distances_{distance_type}_{graph_name}.npy",
                       result_name=f"array_{distance_type}_{graph_name}_cuts{n}_k{k}_imb{imb}.npy",
                       plot_name=f"distrib_{distance_type}_{graph_name}_cuts{n}_k{k}_imb{imb}.png")