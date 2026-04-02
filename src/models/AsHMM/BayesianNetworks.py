import torch as to
from torch import nn


class LGBayesianNetwork(nn.Module):

    def __init__(self,
                 graphs: to.Tensor):
        super(LGBayesianNetwork, self).__init__()
        self.graphs = graphs.int()
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
    
    def my_weights(self, parents: to.Tensor, weight: to.Tensor, aror: to.Tensor, j: int) -> to.Tensor:
        """ extract the weights from the 

        Args:
            parents (to.Tensor): parents of node j
            weight (to.Tensor): weights matrix
            aror (to.Tensor): Ar order vector
            j (int): node 

        Returns:
            to.Tensor: simplified weight vector
        """
        if aror[j] > 0:
            aux = to.Tensor([j, *list(parents), *[self.nfeatures+i for i in range(aror[j])]]).int()
        else:
            aux = to.Tensor([j, *list(parents)]).int()
        return weight[j][aux]


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
        weights: to.Tensor,
        aror: to.Tensor,
        maxar: int,
    ) -> to.Tensor:
        """Computes the means for a timeseries for each feature x depending on the graph, 
        AR orders and weights

        Args:
            x (to.Tensor): input time series
            graph (to.Tensor): graph matrix
            weights (to.Tensor): weights
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
                racum = to.stack([x[maxar - j : -j, k] for j in range(1, aror[k] + 1)]).transpose(0,1)
                acum = to.cat([acum, racum], dim=1)
            weight = self.my_weights(pa_k, weights, aror, k)[None,:]
            mu[:, k] = to.sum(acum * weight, dim=1)
        return mu


    def lg_temp_mu_all(self,
                        x: to.Tensor,
                        weights: to.Tensor,
                        arorder: to.Tensor,
                        maxorder: int)-> to.Tensor:
        """Computes temporal means for a time series for each hidden state

        Args:
            x (to.Tensor): input time series
            weights (to.Tensor): each hidden state network weights  
            arorder (to.Tensor): ar order of each network
            maxorder (int): maximum allowed order

        Returns:
            to.Tensor: temporal mean for each hidden state
        """
        return to.stack([
            self.lg_temp_mu(
                x, self.graphs[i], weights[i], arorder[i], maxorder)
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
                ws = self.my_weights(pas, weights, arord, s)
                s2s = (covi[pas])[:,pas]
                covi[s][s] = sigmas2[s] + (ws[1:len(pas)+1])[None,:] @ s2s @ (ws[1:len(pas)+1])[:,None]
                mui[s] = ws[:len(pas)+1] @ to.cat([to.Tensor([1.]),mui[pas]])
                if arord[s]>0:
                    mui[s] = mui[s]/(1.-to.sum(ws[-arord[s]:]))
                for k in pas:
                    covi[s][k] = to.sum(ws[1: len(pas)+1]*covi[k][pas])
                    covi[k][s] = covi[s][k]
        return [mui, covi]


    def mvn_param_all(self,
                    weights: to.Tensor,
                    sigmas2: to.Tensor,
                    arorder: to.Tensor)-> list:
        """Represent all the linear gaussian network as multivariate normal distributions

        Args:
            weights (to.Tensor): network weights
            sigmas2 (to.Tensor): network variances
            arorder (to.Tensor): network AR orders

        Returns:
            list: list with tuples containing each hidden states' mean vector and covariance matrix
        """
        params = []
        for i in range(self.nstates):
            params.append(
                self.mvn_param(
                    self.graphs[i],
                    self.order[i],
                    weights[i],
                    sigmas2[i],
                    arorder[i]))
        return params

    def forward(self):
        """Dummy forward function

        Returns:
            None
        """
        return None
