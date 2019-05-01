import time
import math
import random
from mct.mcts_env import MCTS_Env1

'''蒙特卡洛搜索树算法
    1.TreeNode为结点类,其state为一个rdkit mol
        get_possible_actions所有可能的action
        is_terminal是否停止,由环境决定
        take_action返回采取动作后的新的结点
        
    2.初始化MCTS类并调用search函数搜索
        resource:包括内存限制(最大模拟次数),时间限制{'time_limit':...,'iter_limit':...}
        initial_state:初始的状态
        rollout_policy:模拟使用的策略,为一函数,传入一个node(也可以是其state)后返回模拟结果
        exploration_constant: 参数,控制exploration的比例
        return: 最后的action(或者是state)
    
    state !!! rollout !!! resource!!!
    '''


#tree node
class TreeNode():
    def __init__(self,state,parent=None,is_terminal=False):
        self.state = state
        self.is_FullyExpanded = is_terminal
        self.parent = parent
        self.num_visit = 0
        self.total_r = 0
        self.children = {}











#MCTS main alogrithm
class MonteCarloTree():
    def __init__(self,config,resource_limit=None,explore_constant=0.2,rollout=None):
        self.resource = resource_limit
        self.eplison = explore_constant
        self.rollout_policy = rollout       #rollout为一个RollOut类的实例,调用step函数进行rollout
        self.env = MCTS_Env1(config)
        self.iter_num = 0



    # input :initial state
    def Search(self,initial_state):

        self.time = time.time()
        self.root = TreeNode(initial_state)
        while self.resource_left():
            leaf = self.tranverse(self.root)
            sim_result = self.rollout_policy.steps(leaf.state)
            self.back_propagate(leaf,sim_result)

        return self.GetAction(self.root,self.GetBestChild(self.root))



    def resource_left(self):
        time_limit = self.time + self.resource['time_limit']
        if time.time()<time_limit and self.iter_num<=self.resource['iter_limit']:
            return True
        else:
            return False


    def back_propagate(self,node,result):
        while node is not None:
            node.num_visit += 1
            node.total_r += result
            node = node.parent



    #遍历子结点以eplison贪心选择结点
    def tranverse(self,node):
        while not self.env.is_terminal(node.state):
            if node.is_FullyExpanded:
                node = self.GetBestChild(node, self.eplison)
            else:
                return self.Expand(node)
        self.iter_num += 1
        return node


    def GetAction(self,root,child):
        for action, node in root.children.items():
            if node is child:
                return action

    # get best child
    def GetBestChild(self,node, exploration_value=0.0):
        best_value = float("-inf")
        best_node = []
        for child in node.children.values():
            nodeValue = child.total_r / child.num_visit + exploration_value * math.sqrt(
                2 * math.log(node.num_visit) / child.num_visit)
            if nodeValue > best_value:
                best_value = nodeValue
                best_node = child
        return best_node

    # 结点向下展开,每次选取一个结点返回,注意action必须是hashable的,即必须封装成元组
    def Expand(self,node):
        actions = self.env.get_possible_actions(node.state)
        for action in actions:
            if action not in node.children.keys():
                new_state = self.env.take_action(node,action)
                newnode = TreeNode(new_state, node, self.env.is_terminal(new_state))
                node.children[action] = newnode
                if len(actions) == len(node.children):
                    node.is_FullyExpanded = True
                return newnode

        raise Exception("Should never reach here")


