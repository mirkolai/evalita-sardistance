__author__ = 'mirko'
import csv
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import pickle
import os.path


def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


class Friend_Network(object):
    G=None
    def __init__(self):

        if os.path.isfile('machinelearning/network/FRIEND.pickle'):
            self.G = pickle.load(open('machinelearning/network/FRIEND.pickle', 'rb'))
        else:
            self.G=nx.Graph()
            csvfile = open("../data/FRIEND.csv")
            next(csvfile)  # skip header
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for relation in spamreader:
                self.G.add_edge(relation[0],relation[1],weight=1)
            print(self.G.order())
            print(self.G.size())
            communities=greedy_modularity_communities(self.G, weight='weight')
            set_node_community(self.G,communities)
            pickle.dump(self.G,open('machinelearning/network/FRIEND.pickle', 'wb'))


    def get_network_community(self,user_id):
        if user_id in self.G:
            return self.G.nodes[user_id]['community']
        else:
            return None


