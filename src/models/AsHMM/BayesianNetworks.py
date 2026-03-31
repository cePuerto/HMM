import torch as to
from torch import nn


class LGBayesianNetwork(nn.Module):
 
    def __init__(self,
                 graphs: to.Tensor,
                 arorders: to.Tensor | None = None,
                 maxorder: int = 0,
                 weights: list | None = None,
                 sigmas2: to.Tensor | None = None):
        super(LGBayesianNetwork, self).__init__()
        self.graphs = graphs
        self.arorders = arorders
        self.maxorder = maxorder
        self.weights = weights
        self.sigmas2 = sigmas2
        nstates, nfeatures, _ = graphs.shape
        self.nstates = nstates
        self.nfeatures = nfeatures
        salidas = self.dag_v_all()
        topor = to.zeros([nstates, nfeatures])
        for i, salida in enumerate(salidas):
            if salida[0] is False:
                raise RuntimeError(f'Error: graph at index {i} is not DAG')
            else:
                topor[i] = salida[1]
        self.order = topor.int()


    def prior_graph(self) -> to.Tensor:
        """ Generates an empty prior graph for each hidden state

        Returns:
            to.Tensor: tensor representing empty BN for each state
        """
        return to.zeros([self.nstates, self.nvariables, self.nvariables], dtype=to.int)


    def my_parents(self, graph: to.Tensor, j: int) -> to.Tensor:
        """ Returns the parents of a node of a graph

        Args:
            graph (to.Tensor): graph tensor
            j (int): index

        Returns:
            to.Tensor: list of parents of index j
        """
        index = to.where(graph[j] == 1)[0]
        return to.sort(index).values


    def dag_v(self, graph: to.Tensor) -> list[bool, to.Tensor]:
        """ Executes the Kahn topological sorting  algorithm.

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


    def dag_v_all(self)->list:
        """Determines for each hidden states if the corresponding graph is dag or not

        Returns:
            list: is a DAG? and its topological order for each graph
        """
        results = []
        for i in range(self.nstates):
            results.append(self.dag_v(self.graphs[i]))
        return results


    def lg_temp_mu(
        self,
        x: to.Tensor,
        graph: to.Tensor,
        weigths: to.Tensor,
        aror: to.Tensor,
        maxar: int,
    ) -> to.Tensor:
        """Computes the means for a timeseries for each feature x depending on the graph, 
        AR orders and weights

        Args:
            x (to.Tensor): input time series
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


    def lg_temp_mu_all(self, x: to.Tensor)-> to.Tensor:
        """ Computes temporal means for a time series for each hidden state

        Args:
            x (to.Tensor): input time series

        Returns:
            to.Tensor: temporal means for each hidden state
        """
        assert self._weights is not None
        assert self._arorders is not None
        assert to.max(self.arorders) <= self.maxorder
        return to.stack([
            self.lg_temp_mu(
                x, self.graphs[i], self.weights[i], self.arorders[i], self.maxorder)
                for i in range(self.nstates)])


    def mvn_param(self,
                  graph: to.Tensor,
                  topor: to.Tensor,
                  weights: to.Tensor,
                  sigmas2: to.Tensor,
                  arord : to.Tensor)-> list:
        """ Returns paramaters of the linear gaussian network as a multivariate normal distribution

        Args:
            graph (to.Tensor): graph
            topor (to.Tensor): topological order
            weights (to.Tensor): weights for the lgbn
            sigmas2 (to.Tensor): variances for the lgbn
            arord (to.Tensor): AR orders for the lgbn

        Returns:
            list: [mean vector, covariance matrix]
        """
        mui = to.zeros([self.nfeatures])
        covi = to.zeros([self.nfeatures, self.nfeatures])
        for s in topor:
            pas = self.my_parents(graph, s)
            if len(pas) == 0:
                mui[s] = weights[s][0]
                covi[s][s] = sigmas2[s]
                if arord[s]>0:
                    mui[s] = mui[s]/(1.-to.sum(weights[s][-arord[s]:]))
            else:
                ws = weights[s]
                s2s = (covi[pas])[:,pas]
                covi[s][s] = sigmas2[s] + (ws[1:len(pas)+1])[None,:] @ s2s @ (ws[1:len(pas)+1])[:,None]
                mui[s] = ws[:len(pas)+1] @ to.cat([to.Tensor([1.]),mui[pas]])
                if arord[s]>0:
                    mui[s] = mui[s]/(1.-to.sum(ws[-arord[s]:]))
                for k in pas:
                    covi[s][k] = to.sum(ws[1: len(pas)+1]*covi[k][pas])
                    covi[k][s] = covi[s][k]
        return [mui, covi]


    def mvn_param_all(self)-> list:
        """Represent all the linear gaussian network as multivariate normal distributions

        Returns:
            list: list with all vectors means and covariance matrices for each state
        """
        assert self.weights is not None
        assert self.arorders is not None
        assert to.max(self.arorders) <= self.maxorder
        params = []
        for i in range(self.nstates):
            params.append(
                self.mvn_param(
                    self.graphs[i],
                    self.order[i],
                    self.weights[i],
                    self.sigmas2[i],
                    self.arorders[i]))
        return params

    def forward(self):
        """Dummy forward function

        Returns:
            None
        """
        return None
