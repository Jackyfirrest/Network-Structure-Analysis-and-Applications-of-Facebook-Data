from functools import cached_property
from itertools import islice, count
from pickletools import int4
from pydoc import pager
import networkx as nx
import pandas
import scipy
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import eigsh

class Centrality:
    def __init__(self, df:pandas.DataFrame) :
        print('Graph created')
        self.G = nx.Graph()

        self.G.add_nodes_from(df.user_id.unique(), bipartite = 0)
        self.G.add_nodes_from(df.page_id.unique(), bipartite = 1)

        mask = df.reaction_time != 0
        df = df[mask]

        print('Creating edges...')
        self.G.add_weighted_edges_from(
            zip(df.user_id, df.page_id, df.reaction_time)
            )
        print('Graph creation completed successfully')

        self.user_node = sorted({n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0})
        self.page_node = sorted({n for n, d in self.G.nodes(data=True) if d["bipartite"] == 1})


    @cached_property
    def degree_centrality(self):
        """
        Returns the degree_centrality of the graph.
        """
        return nx.bipartite.degree_centrality(self.G, self.user_node)
        
    def user_centrality(self, centrality_dict:dict, reverse:bool=True) -> dict:
        """
        Returns the users' centrality given the centrality dictionary.
        """
        user_c_unsorted = {x:centrality_dict[x] for x in self.user_node}
        return dict(
            sorted( user_c_unsorted.items(), key=lambda item:item[1], reverse = reverse)
            )

    def page_centrality(self, centrality_dict:dict, reverse:bool=True) -> dict:

        page_c_unsorted = {x:centrality_dict[x] for x in self.page_node}
        return dict(
            sorted( page_c_unsorted.items(), key=lambda item:item[1], reverse = reverse)
            )
            
    @cached_property
    def A_matrix(self):
        """
        The biadjacency matrix of the graph. It is n_u * n_p, where n_u is the number of user_nodes and n_p is the number of page_nodes.

        The value of the elements are the weight of the graph.
        """
        return nx.bipartite.biadjacency_matrix(
                self.G, row_order=self.user_node, column_order=self.page_node)

    @cached_property
    def A_matrix_no_weight(self):
        return (self.A_matrix > 0) * 1

    @cached_property
    def Social_matrix(self):
        """
        Build a social matrix from the A_matrix.

        The A_matrix is n_u * n_p, where n_u is the number of users and n_p is the number of pages.

        The Social_matrix is [[0, A],[A', 0]], and therefore the columns(and rows) represent [users, pages]
        """
        return bmat([[None, self.A_matrix],[self.A_matrix.transpose(), None]])

    @cached_property
    def page_comembership_matrix(self):
        """
        The unweighted co-membership matrix of the pages.

        It is defined as A'A, where A is the biadjacency matrix of the graph.

        Return
        ------
        A m-by-m matrix, where m is the number of pages in the graph.
        """
        A = self.A_matrix_no_weight
        return (A.transpose() @ A).asfptype()

    @cached_property
    def user_comembership_matrix(self):
        """
        The unweighted co-membership matrix of the users.

        It is defined as AA', where A is the biadjacency matrix of the graph.

        Returns
        -------
        A n-by-n matrix, where n is the number of users in the graph.
        """
        A = self.A_matrix_no_weight
        return (A @ A.transpose()).asfptype()


    @cached_property
    def page_comembership_graph(self):

        g = nx.from_scipy_sparse_matrix(self.page_comembership_matrix)
        mapping = {u:v for u, v in zip(count(), self.page_node)}
        return nx.relabel_nodes(g, mapping=mapping)

    def eigenvector_centrality_func(self, weight = 'weight'):
        """
        Returns the eigenvector centrality of the graph for users and pages, respectively.

        The calculation might be time consuming.

        However, we only have to calculate ones. Because of the property
        Ac^U = \lambda c^P
        A'c^P = \lambda c^U

        Once we get the eigenvector and eigenvalue of pages, which is only 1000*1000, we can calculate the eigenvectors of the users directly from it.
        """
        A = self.A_matrix_no_weight
        if weight == 'weight':
            A = self.A_matrix

        print("Calculating pages' eigenvector centrality")
        evalue_page, evector_page = eigsh((A.transpose() @ A).asfptype(), k=1, which='LA') 
        
        evalue_page = evalue_page[0]

        print("Calulating user's eigenvector centrality")
        evector_user = (A @ evector_page) / np.sqrt(evalue_page)

        evector_page = dict(zip(self.page_node,[abs(r[0]) for r in evector_page]))
        evector_user = dict(zip(self.user_node,[abs(r[0]) for r in evector_user]))

        return evector_user, evector_page

    @cached_property
    def eigenvector_centrality(self):
        return self.eigenvector_centrality_func('weight')

    @cached_property
    def eigenvector_centrality_unweighted(self):
        return self.eigenvector_centrality_func(weight = 'None')

    ## without weight?
    @cached_property
    def closeness_centrality(self):
        # return nx.bipartite.closeness_centrality(self.G, self.user_node)
        normalized = True
        closeness = {}
        path_length = nx.single_source_shortest_path_length
        top = set(self.page_node)
        bottom = set(self.user_node)
        n = len(top)
        m = len(bottom)

        print('Processing closeness centrality')
        for i, node in enumerate(top):
            print(f'Processing {i+1}/{n}', end = "\x1b\r")
            sp = path_length(self.G, node)
            totsp = sum(sp.values())
            if totsp > 0.0:
                s = (len(sp) - 1) / (len(self.G) - 1)
                closeness[node] = (m + 2 * (n - 1)) * s / totsp
            else:
                #closeness[n] = 0.0
                closeness[node] = 0.0
        # for node in bottom:
        #     sp = dict(path_length(G, node))
        #     totsp = sum(sp.values())
        #     if totsp > 0.0 and len(G) > 1:
        #         closeness[node] = (n + 2 * (m - 1)) / totsp
        #         if normalized:
        #             s = (len(sp) - 1) / (len(G) - 1)
        #             closeness[node] *= s
        #     else:
        #         #closeness[n] = 0.0
        #         closeness[node] = 0.0
        print()
        return closeness


    @cached_property
    def community_detection(self):
        return nx.community.greedy_modularity_communities(self.page_comembership_graph, weight = 'weight')

    @staticmethod
    def combine_centralities(result_path:str, file_type:str = 'csv'):
        import os
        file_list = [file for file in os.listdir(result_path) if file.endswith("." + file_type)]
        week_list = [extract_week(week) for week in file_list]

        df = pandas.DataFrame()

        for week, file in zip(week_list,file_list):
            newdf = pandas.read_csv(f"{result_path}/{file}")
            newdf["eigenvector_centrality"] = abs(newdf["eigenvector_centrality"])
            newdf["unweighted_eigenvector_centrality"] = abs(newdf["unweighted_eigenvector_centrality"])

            newdf['week'] = week
            df = df.append(newdf, ignore_index=True)

        df.to_csv(f'{result_path}/Full_centrality.csv')

        return df
    
    


def topn(d:dict, n:int) ->dict:
    """
    Get the top 10 elements of the dictionary from the given dictionary.
    """
    return dict(islice(d.items(), 0, n))

def extract_week(s:str)->str:
    return s.split('_')[0]

from dataclasses import dataclass
from typing import Dict

@dataclass
class CommunityStat:
    week: str
    trump_community:list
    clinton_community:list
    other_community:list

    def filter_by_centrality(self, centrality_list):
        self.trump_community = [id for id in self.trump_community if id in centrality_list]
        self.clinton_community = [id for id in self.clinton_community if id in centrality_list]
        self.other_community = [id for id in self.other_community if id in centrality_list]
        return self
    
    @property
    def dict(self):
        self.__dict = dict()
        self.__dict.update({t:1 for t in self.trump_community})
        self.__dict.update({c:2 for c in self.clinton_community})
        self.__dict.update({o:3 for o in self.other_community})

        return self.__dict

    def average_centrality(self, week_centrality:Dict):
        """
            week_centrality : {
                182739503862  : 0.73,
                9672727986293 : 0.23,...
            }
        """
        import numpy as np
        trump_centrality = np.mean([week_centrality[id] for id in self.trump_community ])
        clinton_centrality = np.mean([week_centrality[id] for id in self.clinton_community ])
        other_centrality = np.mean([week_centrality[id] for id in self.other_community ])

        return (trump_centrality, clinton_centrality, other_centrality)