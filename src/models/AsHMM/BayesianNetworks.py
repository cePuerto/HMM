import torch as to

def prior_graph(nstates : int ,nvariables: int) -> to.Tensor:
    """Generates an empty prior graph for each hidden state 

    Args:
        nstates (int): Number of hidden states
        nvariables (int): Number of features 

    Returns:
        to.Tensor: int8 tensor representing empty BN for each state
    """
    return to.zeros([nstates,nvariables,nvariables], dtype=to.int8)


def dag_v(graph: to.Tensor) -> list[bool,to.Tensor]:
    """Executes the Kahn topological sorting  algorithm. 

    Args:
        graph (to.Tensor): int8 tensor of size [nfeatures, nfeatures] representing a BN

    Returns:
        list[bool, to.Tensor]: is a DAG? and its topological order
    """
    graph_copy = to.copy(graph)
    topor = []
    booli = False
    ss = to.sum(graph_copy,dim=1)
    s = to.where(ss==0)[0]
    while len(s)>0:
        si = s[0]
        s = s[1:]
        topor.append(si)
        indexi = to.where(graph_copy.T[si]==1)[0]
        for x in indexi:
            graph_copy[x][si] = 0
            if to.sum(graph_copy[x])==0:
                s =to.concatenate([s,[x]])
    if to.sum(graph_copy)==0:
        booli = True
    return [booli,to.Tensor(topor,dtype=to.int8)]
