import networkx as nx

#基类定义的简单些 就一个sample函数 其中nx4mol是一个nx的对象，这个nx的对象用以表示一个分子
#sample函数的返回要求是一个三元组<adj,node_array,action>其中action又是一个4元组，具体参考以前的文档，其中最后一个分量terminal，用0表示继续，用1表示停止
#我们先只考虑qm9的情况，zinc的任务之后再做，到时候肯定要修改代码的
class SampleTrajectory(object):
    def __init__(self):
        pass
    
    def sample(self, nx4mol):
        pass