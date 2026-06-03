import torch as to
from torch import nn
from torch.linalg import solve
from src.pydant import ModelGeneralConfig
from src.models.AsHMM.BayesianNetworks import LGBayesianNetwork as lgb
from src.models.forback import ForwardBackward

class AsHMM(nn.Module):

    def __init__(self, config: ModelGeneralConfig):
        super(AsHMM, self).__init__()
        self.config = config
        self.transition = to.ones([config.nhidden, config.nhidden])/config.nhidden
        self.initial = to.ones(config.nhidden)/config.nhidden
        self.graphs = to.zeros([config.nhidden, config.nfeatures, config.nfeatures]).int()
        self.arorders = to.zeros([config.nhidden, config.nfeatures]).int()
        self.sigma2 = to.ones([config.nhidden, config.nfeatures])
        self.weights = to.zeros([config.nhidden, config.nfeatures, config.nfeatures + config.arorder])
        self.nstates = config.nhidden
        self.marorder = config.arorder
        self.nfeatures = config.nfeatures
        self.nparams = 0
        self.lgnetworks = lgb(self.graphs)
        self.relevancies = config.saliencies
        self.myforback = []


    def init_params(self, x :to.Tensor, cuts: to.Tensor):
        """Given an input, it creates strating search points for weights and standard deviations
        this method is required to activate weights' and variances' autograd property

        Args:
            x (to.Tensor): input sequences
            cuts(to.tensor): cut points for the input sequences
        """
        maximum = to.amax(x, dim=0)
        minimum = to.amin(x, dim=0)
        means =  [minimum + (i+1)*(maximum-minimum)/(self.nstates+2) for i in range(self.nstates)]
        sigmas2 = to.Tensor([to.var(x[:,i]) for i in range(self.nfeatures)])
        for i in range(self.nstates):
            self.weights[i,:self.nfeatures, :self.nfeatures] = to.diag(means[i])
            self.sigma2[i] = sigmas2
        self.nparams = self.nstates*(1+self.nstates+self.nfeatures)+ to.sum(self.graphs).item()
        self.myforback = [ForwardBackward(self.nstates) for _ in range(len(cuts)-1)]


    def penalty(self,graph: to.Tensor, arorders: to.Tensor, length: int)->list:
        """Calculates the Bayesian informaticon criterion BIC penalty for the provided graph

        Args:
            graph (to.Tensor): Bayesian network  
            arorders (to.Tensor): AR orders
            length (int): lengths

        Returns:
            list: [BIC penalty, Number parameters]
        """
        b= to.sum(graph)+ self.nstates*(1 + self.nfeatures + self.nstates)+to.sum(arorders)
        return [-b*0.5*to.log(length),b.int()]


    def compute_mut_all(self, x: to.Tensor, cuts: list[int], graphs: to.Tensor, weights: to.Tensor, arorders: to.Tensor)-> list[to.Tensor]:
        """Computes temporal means for each input time series divided by cut

        Args:
            x (to.Tensor): concatenated time series 
            cuts (list[int]): cuts determining the start and end of each time series eg. [0, len(x)] for a single time series
            [0,len(x_1),len(x1)+len(x2)] for two time series

        Returns:
            list[to.Tensor]: returns a list with tensors with the temporal mean for each time series
        """
        self.lgnetworks.update_graphs(graphs)
        return [self.lgnetworks.lg_temp_mu_all(
            x[cuts[i]:cuts[i+1]],
            weights,
            arorders,
            self.marorder) for i in range(len(cuts)-1)]


    def compute_probt(self, x: to.Tensor, tmu: to.Tensor, sigma2: to.Tensor, reduction : list[int] = 2)->to.Tensor:
        """Computes the log-likelihood of each feature for each hidden state

        Args:
            x (to.Tensor): input time series
            tmu (to.Tensor): temporal mean for each hidden states, time instance and feature
            sigma2 (to.Tensor): Variances matrix
            reduction (list[int]|None): Reduction using sum for the output 
            prob of shape [nstates,length,nfeatures], if None, no reduction is performed. Default: [2]

        Returns:
            to.Tensor: log-likelihoods for the specified reduction 
        """
        if reduction is not None:
            return to.sum(-0.5 * (to.log(2. * to.Tensor([to.pi]))+ to.log(sigma2)[:,None,:]+
                        ((x[self.marorder:])[None,:]-tmu)**2/self.sigma2[:,None,:]),dim= reduction)
        else:
            return -0.5 * (to.log(2. * to.Tensor([to.pi]))+ to.log(sigma2)[:,None,:]+
                        ((x[self.marorder:])[None,:]-tmu)**2/sigma2[:,None,:])


    def compute_probt_all(self, x: to.Tensor, cuts: list[int], meant : list, sigma2: to.Tensor, reduction : list[int] = 2) -> list[to.Tensor]:
        """Computes the temporal log-likelihood for each time series 

        Args:
            x (to.Tensor): concatenated time series 
            cuts (list[int]): cuts determining the start and end of each time series eg. [0, len(x)] for a single time series
            [0, len(x_1),len(x1)+len(x2)] for two time series
            meant (list): temporal means 
            sigma2 (to.Tensor): Variances matrix
            reduction (list[int]|None): Reduction using sum for the output 
            prob of shape [nstates,length,nfeatures], if None, no reduction is performed. Default: [2]

        Returns:
            list[to.Tensor]: temporal log-likelihood for each input time series
        """
        return [self.compute_probt(x[cuts[i]:cuts[i+1]],meant[i], sigma2, reduction) for i in range(len(cuts)-1)]


    def compute_gamma_all(self, x: to.Tensor, cuts: to.Tensor, graphs: to.Tensor, weights: to.Tensor, sigma2: to.Tensor, arorders: to.Tensor) ->list:
        """Computes all the gamma statistics for all the time series

        Args:
            x (to.Tensor): concatenated time series 
            cuts (to.Tensor): cuts  for input time series
            graphs (to.Tensor): tensor representing the graphs
            weights (to.Tensor): tensor representing the weights
            sigma2 (to.Tensor): Variances matrix
            arorders (to.Tensor): tensor representing the AR orders

        Returns:
            list: [all temporal means, all temporal full probabilities
        """
        meant_all = self.compute_mut_all(x, cuts, graphs, weights, arorders)
        prob_list = self.compute_probt_all(x, cuts, meant_all, sigma2, reduction=None)
        for i, probt_full  in enumerate(prob_list):
            self.myforback[i].compute_gamma(self.initial, self.transition, probt_full)
        return meant_all, prob_list


    def collect_statistics(self, x: to.Tensor, cuts: to.Tensor) -> list:
        """Computes latent probabilities and statistics to update parameters

        Args:
            x (to.Tensor): input timeseries
            cuts (to.Tensor): cuts for the timeseries
        """
        meant_all = self.compute_mut_all(x, cuts, self.graphs, self.weights, self.arorders)
        prob_list_full = self.compute_probt_all(x, cuts, meant_all, self.sigma2, reduction=None)
        self.myforback[0].compute_gamma(self.initial, self.transition, prob_list_full[0])
        s_p = self.myforback[0].act_initial()
        s_a = self.myforback[0].act_transition(self.transition, to.sum(prob_list_full[0],dim = 2))
        s_b = self.myforback[0].act_weights_ashmm(x, self.graphs, self.arorders, self.marorder)
        s_v = self.myforback[0].act_sigma2_ashmm(x, meant_all[0], self.marorder)
        loglikelihood = self.myforback[0]()
        for i in range(1,len(prob_list_full)):
            self.myforback[i].compute_gamma(self.initial, self.transition, prob_list_full[i])
            s_p += self.myforback[i].act_initial()
            s_a += self.myforback[i].act_transition(self.transition, to.sum(prob_list_full[i],dim2 = 2))
            s_b += self.myforback[i].act_weights_ashmm(x, self.graphs, self.arorders, self.marorder)
            s_v += self.myforback[i].act_sigma2_ashmm(x, meant_all[i], self.marorder)
            loglikelihood += self.myforback[i]()
        return [[s_p, s_a, s_b, s_v], loglikelihood]


    def update_transition(self, s_a: list)-> to.Tensor:
        """Returns an updated transition probability matrix

        Args:
            s_a (list): list with statistics to update the transition matrix

        Returns:
            to.Tensor: an updated transition matrix
        """
        numa, dena = s_a
        return numa/dena


    def update_initial(self, s_p : list, n_series: int) -> to.Tensor:
        """Returns the updated initial distribution

        Args:
            s_p (list): statistical for initial distribution 
            n_series (int): number of time series

        Returns:
            to.Tensor: updated initial distribution
        """
        return s_p/n_series


    def update_weights(self, s_b: list)-> to.Tensor:
        """Updates the weight aprameter updated

        Args:
            s_b (list): statistics to update weights

        Returns:
            to.Tensor: updated weights 
        """
        bc , ac = s_b
        nweight = to.zeros([self.nstates, self.nfeatures, self.nfeatures+self.marorder])
        for i in range(self.nstates):
            for k in range(self.nfeatures):
                if bc[i][k].shape.numel() > 1:
                    bik = solve(bc[i][k],ac[i][k])
                else:
                    bik = (ac[i][k]/bc[i][k])[0]
                pa_ik = lgb.my_parents(self.graphs[i],k)
                nweight[i][k][k] = bik[0]
                for j, pa in enumerate(pa_ik): 
                    nweight[i][k][pa] = bik[j+1]
                for j in range(self.arorders[i][k]):
                    nweight[i][k][self.nfeatures+j] = bik[1+len(pa_ik)+j]

        return nweight


    def update_sigma2(self, s_v : list)-> to.Tensor:
        """Returns an updated sigma2 parameter

        Args:
            s_v (list): statistics to update sigma2 parameter

        Returns:
            to.Tensor: updated sigma2 parameter
        """
        nums, dens = s_v
        return to.clip(nums/dens,1e-5)


    def update_all(self, s_p: to.Tensor, s_a: list, s_b: list, s_v: list, nseq: int):
        """Updates all parameters

        Args:
            s_p (to.Tensor): statistics to update initial distribution
            s_a (list): statistics to update transition matrix
            s_b (list): statistics to update weight parameter
            s_v (list): statistics to update sigma2 parameter
            nseq (int): number of time series
        """
        self.transition = self.update_transition(s_a)
        self.initial = self.update_initial(s_p, nseq)
        self.weights = self.update_weights(s_b)
        self.sigma2 = self.update_sigma2(s_v)


    def compute_EM(self, x: to.Tensor, cuts: to.Tensor, reset : bool = False):
        """Performs the EM algorithm for a fixed graph and AR-order

        Args:
            x (to.Tensor): input sequences
            cuts (to.Tensor): cuts of the time series
            reset (bool): whether to reset the model parameters
        """
        if reset or len(self.myforback) == 0:
            self.init_params(x, cuts)
        nseq = len(cuts)-1
        [stats,llike] = self.collect_statistics(x, cuts)
        error = 1e10
        it = 0
        while (error > self.config.training.epsilon and it < self.config.training.nepochs):
            self.update_all(*stats, nseq)
            [stats, nllike] = self.collect_statistics(x, cuts)
            error = to.abs(nllike-llike)
            print(f"Iteration : {it}, error: {error}, ll: {nllike}")
            llike = nllike
            it+=1


    def local_score(self, pena: float, target_state: int, target_feature: int)->to.Tensor:
        """ Computes local score for a given state 

        Args:
            probt (to.Tensor): Tensor of full probs [nsequences, nstates, length, nfeatures]
            pena (float): Penalization
            target_state (int): modified state
            target_feature (int): modified feature

        Returns:
            to.Tensor: local score fro SEM optimization
        """
        local_s = 0.
        for forback in self.myforback:
            local_s += to.sum(forback.gamma[:,target_state]*forback.temporal_prob_full[target_state,:,target_feature])
        return local_s+pena
    

    def score_complete_info_estimation_local(
            self, 
            x : to.Tensor, 
            cuts : to.Tensor,
            prob_list: to.Tensor,
            meant_all : to.Tensor,
            graph : to.Tensor,
            ar_orders : to.Tensor,
            state: int,
            feature: int) -> list:
        """ COmputes local scores for a given state and feature

        Args:
            x (to.Tensor): input sequence
            cuts (to.Tensor):  cuts of the time series
            prob_list (to.Tensor): probabilities for each state and feature
            meant_all (to.Tensor): means of all states and features
            graph (to.Tensor): graph to be used
            ar_orders (to.Tensor): autoregressive orders
            state (int): hidden state to be updated
            feature (int): feature to be updated

        Returns:
            list: [local score, new weights, new variance matrix]
        """
        pena = self.penalty(graph, ar_orders, cuts[-1])
        self.myforback.compute_gamma(prob_list[0])
        s_b = self.myforback.act_weights_ashmm(x, graph, ar_orders, self.marorder)
        s_v = self.myforback.act_sigma2_ashmm(x, meant_all[0], self.marorder)
        for i in range(1,len(prob_list)):
            self.myforback.compute_gamma(prob_list[i])
            s_b += self.myforback.act_weights_ashmm(x, self.graphs, self.arorders, self.marorder)
            s_v += self.myforback.act_sigma2_ashmm(x, meant_all[i], self.marorder)
        nweight = self.update_weights(s_b)
        nvar = self.update_sigma2(s_v)
        nmut = self.compute_mut_all(x, cuts, nweight,ar_orders)
        nprob = self.compute_probt_all(x,cuts, nmut, nvar,  None)
        ll = self.local_score(nprob, pena , state, feature)
        return [ll,nweight,nvar]


    def climb_ar(self, x : to.Tensor, cuts: list[int]):
        """
        Looks for the best structure in AR components. 
        Uses a greedy search
        """
        for i in range(self.nstates):
            for k in range(self.nfeatures):
                base_pen = self.penalty(self.graphs, self.arorders, cuts)
                sm = self.local_score(base_pen, i, k)
                while self.arorders[i][k] +1 <= self.marorder :
                    arord2 = to.clone(self.arorders)
                    arord2[i][k] = arord2[i][k] +1
                    [s2,nweight,nsigma2] = self.score_complete_info_estimation_local(
                        x, cuts, probt_all, mut_all, self.graphs, arord2, i, k
                    )
                    if s2 > sm:
                        self.arorders = arord2
                        self.weights = nweight
                        self.sigma2 = nsigma2
                        sm  = s2
                    else:
                        break


    def pos_ads(self, graph: to.Tensor) -> list:
        """ Looks for possible arcs to be added to the graph

        Args:
            graph (to.Tensor): graph representation

        Returns:
            list: A list where
        list[.][0] is a node which can recieve edges
        list[.][1] is a list  of potential fathers
        """
        index = []
        for i in range(self.nfeatures):
            indexi = [i]
            indexj =[]
            for j in range(self.nfeatures):
                ngraph = to.clone(graph)
                if ngraph[i][j] != 1:
                    ngraph[i][j] = 1
                    [fool, _] = self.lgnetworks.dag_v(ngraph)
                    if fool  is True:
                        indexj.append(j)
            indexi.append(indexj)
            index.append(indexi)
        return index
    

    # def climb_struc(self,tb,ts): 
    #     """
    #     Looks for the best graphical structure, uses an upward greedy algorithm

    #     Parameters
    #     ----------
    #     tb : TYPE boolean list 
    #         DESCRIPTION. indices to be updated of the parameter B
    #     ts : TYPE boolean list
    #         DESCRIPTION. indices to be updated of the parameter sigma
    #     """
    #     for i in range(self.N):
    #         if tb[i] ==1 or ts[i]==1 :
    #             for k in range(self.K):
    #                 possi = self.pos_ads(self.G[i])
    #                 son = possi[k][0]
    #                 if len(possi[k][1])!=0:
    #                     sm = self.local_score(self.G,self.p,self.sigma,i,son)[0]
    #                     for j in possi[k][1]:
    #                         G2 = np.copy(self.G)
    #                         G2[i][son][j] =1
    #                         L2 = []
    #                         for nn in range(self.N):
    #                             L2.append(self.dag_v(G2[nn])[1])
    #                         [s2,B2,sigma2,b2] = self.score_complete_info_estimation_local(G2,self.L,self.p,i,son,tb,ts)
    #                         if s2>sm: # Teorema: Irresoluble
    #                             sm= s2
    #                             self.B = B2
    #                             self.b = b2
    #                             self.sigma = sigma2
    #                             self.G = G2
    #                             self.L = L2

                                
    #     self.act_mut(self.G,self.B,self.p)


    def hill_climb(self, lags : bool, struct: bool, x: to.Tensor, cuts: to.Tensor):
        """Executes a greedy search to add nodes to the graphs

        Args:
            lags (bool): add temporal nodes?
            struct (bool): add inter-temporal nodes?
            x (to.Tensor): input sequences
            cuts (to.Tensor): cut points for x
        """
        if lags is True:
            self.climb_ar(x, cuts)
        # if struct is True:
        #     self.climb_struc()


    def SEM(self, err1=1e-2,err2=1e-2,its1=100,its2=100): 
        """
        Does the SEM algorithm for parameter and structure learning

        Parameters
        ----------
        err1 : TYPE, optional float
            DESCRIPTION. The default is 1e-2. Maximum SEM error allowed
        err2 : TYPE, optional float
            DESCRIPTION. The default is 1e-2. Maximum EM error allowed
        its1 : TYPE, optional int
            DESCRIPTION. The default is 100. Maximum SEM iterations
        its2 : TYPE, optional int
            DESCRIPTION. The default is 100. Maximum EM iterations 
        ta : TYPE, optional  boolean list
            DESCRIPTION. The default is "all".  list that gives the indices of rows and columns of A to be updated.
        tpi : TYPE, optional bool
            DESCRIPTION. The default is True. Updates or not the initial distribution
        tb : TYPE, optional boolean list
            DESCRIPTION. The default is "all". indices to be updated of the parameter B
        ts : TYPE, optional boolean list
            DESCRIPTION. The default is "all". list that gives the indices of which hidden-states variances are updated 
        ps : TYPE, optional int
            DESCRIPTION. The default is None. index of the initial distibution
        """
        eps = 1
        it = 0
        likelihood = [ ]
        self.EM(its2,err2,ta=ta,tb=tb,tpi=tpi,ts=ts,ps=ps)
        likelihood.append(self.bic)
        while eps > err1 and it < its1:
            self.hill_climb(tb,ts)
            self.EM(its2,err2,ta=ta,tb=tb,tpi=tpi,ts=ts,ps=ps)
            eps = np.abs((self.bic-likelihood[-1]))
            likelihood.append(self.bic)
            it = it+1
       
