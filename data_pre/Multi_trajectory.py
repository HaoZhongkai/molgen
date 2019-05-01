import networkx as nx
from Funcs import nx2adj, nx2node, mol2nx,adj_add_scaffold,nx2mol
import random
from copy import deepcopy as dcopy
import numpy as np
import rdkit.Chem as Chem

'''对每个分子的生成过程同时使用多种采样方式进行采样'''

class Multi_trajectory():
    def __init__(self,config):
        self.max_atom_num = config.max_atom_num
        self.possible_bonds = config.possible_bonds
        self.max_iter = config.max_tr_per_molecule
        self.possible_atoms = config.possible_atoms

        self.sample__ = self.random_sample


    # return a list of <adj, node_array, action>
    def sample(self, nx4mol):
        #获得所有的C原子的list
        #将mol转化为graph nx
        mol_nx_graph = mol2nx(nx4mol)
        C_list = []
        for node in mol_nx_graph.nodes:


            if mol_nx_graph.nodes[node]['atomic_num'] is 6:
                C_list.append(node)


            if len(C_list)>=self.max_iter:
                break


        #对每个C做一次随机图搜索,并搜集采样

        trajectory = []
        mol_trajectory = []


        iterate = 1
        for node in C_list:
            if iterate> self.max_iter:
                break
            sample_tra, mol_graph = self._to_action(self.sample__(mol_nx_graph,node))

            trajectory = trajectory + sample_tra
            mol_trajectory = mol_trajectory + mol_graph


        return trajectory,mol_trajectory



    # convert a state graph to trajectory:(adj, node, action)
    def _to_action(self,state_graph):
        graph_list = []
        trajectory = []
        for index in range(len(state_graph)):

            graph, start, end, cycle = state_graph[index]
            adj = adj_add_scaffold(nx2adj(graph, self.max_atom_num, self.possible_bonds),self.possible_atoms)
            node_array = nx2node(graph,self.max_atom_num,self.possible_atoms)


            '''for test'''
            graph_list.append(graph)

            #加环情况
            if index==len(state_graph)-1:
                stop = 1
            else:
                stop = 0

            bond_type = graph.get_edge_data(start, end)['bond_type']


            start_one_hot = np.zeros([self.max_atom_num])
            end_one_hot = np.zeros([self.max_atom_num+len(self.possible_atoms)])
            bond_one_hot = np.zeros([len(self.possible_bonds)])
            bond_one_hot[self.possible_bonds.index(bond_type)] = 1
            start_one_hot[start] = 1


            if cycle:
                end_one_hot[end] = 1
                action = [start_one_hot, end_one_hot,bond_one_hot, stop]
            else:
                #  这里需要原子的类型来检测action的值
                end_atom_pos = self.max_atom_num+self.possible_atoms.index(Chem.Atom(
                    graph.nodes[end]['atomic_num']).GetSymbol())
                end_one_hot[end_atom_pos] = 1
                action = [start_one_hot, end_one_hot,bond_one_hot, stop]



            trajectory.append((adj, node_array, action))
        return trajectory,graph_list



    # return a action by random sampling
    # input : G nx graph
    # return [(graph, start, end, cycle)] start,end 为当前图在前一步的基础上加的边
    def random_sample(self, G, root):
        '''采样,随机生成产生分子的路径,'''
        ### specially intention that nodes added to new graph should be reordered
        state_graph = []

        V = nx.Graph()
        V.add_node(0,**G.nodes[root])


        # S是所有与V有边相连的边的集合(v,u),其中v在V种,u在S中
        S = [(root, u) for u in G.neighbors(root)]
        #补的node映射表,第一个为原图的结点序,第二个为生成的图的结点序
        order_chart = {root:0}
        while V.number_of_nodes()<G.number_of_nodes():
            new_nodes = random.choice(S)
            start_node, end_node = new_nodes
            order_chart[end_node] = V.number_of_nodes()
            V.add_node(V.number_of_nodes(), **G.nodes[end_node])
            V.add_edge(order_chart[start_node], order_chart[end_node],
                       bond_type=G.get_edge_data(start_node,end_node)['bond_type'])
            #update the graph state
            H = nx.Graph(V)
            state_graph.append((H,order_chart[start_node],order_chart[end_node],0))

            S.remove(new_nodes)

            for w in G.neighbors(end_node):
                if w in order_chart.keys():
                    if not V.has_edge(order_chart[w],order_chart[end_node]):

                        #cycle detect

                        V.add_edge(order_chart[end_node], order_chart[w],
                                   bond_type=G.get_edge_data(end_node,w)['bond_type'])
                        print('cycle_detected')
                        S.remove((w, end_node))
                        state_graph.append((nx.Graph(V),order_chart[w],order_chart[end_node],1))

                else:
                    S.append((end_node, w))


        return state_graph









