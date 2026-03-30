import torch as to
from torch import nn


class BayesianNetwork(nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()

    def prior_graph(self, nstates: int, nvariables: int) -> to.Tensor:
        """Generates an empty prior graph for each hidden state

        Args:
            nstates (int): Number of hidden states
            nvariables (int): Number of features

        Returns:
            to.Tensor: int tensor representing empty BN for each state
        """
        return to.zeros([nstates, nvariables, nvariables], dtype=to.int)

    def my_parents(self, graph: to.Tensor, j: int) -> to.Tensor:
        """_summary_

        Args:
            graph (to.Tensor): graph tensor
            j (int): index

        Returns:
            to.Tensor: list of parents of index j
        """
        index = to.where(graph[j] == 1)[0]
        return to.sort(index).indices

    def dag_v(self, graph: to.Tensor) -> list[bool, to.Tensor]:
        """Executes the Kahn topological sorting  algorithm.

        Args:
            graph (to.Tensor): int tensor of size [nfeatures, nfeatures] representing a BN

        Returns:
            list[bool, to.Tensor]: is a DAG? and its topological order
        """
        graph_copy = to.clone(graph)
        topor = []
        booli = False
        ss = to.sum(graph_copy, dim=1)
        s = to.where(ss == 0)[0]
        while len(s) > 0:
            si = s[0]
            s = s[1:]
            topor.append(int(si))
            indexi = to.where(graph_copy.transpose(0, 1)[si] == 1)[0]
            for x in indexi:
                graph_copy[x][si] = 0
                if to.sum(graph_copy[x]) == 0:
                    s = to.cat([s, to.Tensor([x]).int()])
        if to.sum(graph_copy) == 0:
            booli = True
        return [booli, to.Tensor(topor).int()]

    def linear_gauss_means(
        self,
        x: to.Tensor,
        graph: to.Tensor,
        weigths: to.Tensor,
        aror: to.Tensor,
        maxar: int,
    ) -> to.Tensor:
        """Computes the means for a timeseries for each feature x depending on the graph, AR orders and weights

        Args:
            x (to.Tensor): time series
            graph (to.Tensor): graph matrix
            weigths (to.Tensor): weights
            aror (to.Tensor): vector indicating the ar order for each feature
            maxar (int): maximum lag

        Returns:
            to.Tensor: means for each feature
        """
        length, nfeatures = x.shape
        mu = to.zeros([length - maxar, nfeatures])
        for k in range(nfeatures):
            pa_k = list(self.my_parents(graph, k))
            acum = to.cat([to.ones([length - maxar, 1]), x[maxar:, pa_k]], axis=1)
            if aror[k] > 0:
                racum = to.stack([x[maxar - j : -j, k] for j in range(1, aror[k] + 1)]).T
                acum = to.cat([acum, racum], dim=1)
            mu[:, k] = to.sum(acum * weigths[k][None, :], dim=1)
        return mu
