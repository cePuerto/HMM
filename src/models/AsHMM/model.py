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
        self.lgnetworks = lgb(self.graphs)
        self.relevancies = config.saliencies
        self.myforback = ForwardBackward(self.transition, self.initial, self.nstates)


    def init_params(self, x :to.Tensor):
        """Given an input, it creates strating search points for weights and standard deviations
        this method is required to activate weights' and variances' autograd property

        Args:
            x (to.Tensor): input tensor
        """
        maximum = to.amax(x, dim=0)
        minimum = to.amin(x, dim=0)
        means =  [minimum + (i+1)*(maximum-minimum)/(self.nstates+2) for i in range(self.nstates)]
        sigmas2 = to.Tensor([to.var(x[:,i]) for i in range(self.nfeatures)])
        for i in range(self.nstates):
            self.weights[i,:self.nfeatures, :self.nfeatures] = to.diag(means[i])
            self.sigma2[i] = sigmas2


    def compute_mut_all(self, x: to.Tensor, cuts: list[int])-> list[to.Tensor]:
        """Computes temporal means for each input time series divided by cut

        Args:
            x (to.Tensor): concatenated time series 
            cuts (list[int]): cuts determining the start and end of each time series eg. [0, len(x)] for a single time series
            [0,len(x_1),len(x1)+len(x2)] for two time series

        Returns:
            list[to.Tensor]: returns a list with tensors with the temporal mean for each time series
        """
        return [self.lgnetworks.lg_temp_mu_all(
            x[cuts[i]:cuts[i+1]],
            self.weights,
            self.arorders,
            self.marorder) for i in range(len(cuts)-1)]


    def compute_probt(self, x: to.Tensor, tmu: to.Tensor)->to.Tensor:
        """Computes the log-likelihood of each feature for each hidden state

        Args:
            x (to.Tensor): input time series
            tmu (to.Tensor): temporal mean for each hidden states, time instance and feature

        Returns:
            to.Tensor: log-likelihoods for each hidden state, time instance and feature
        """
        return to.sum(-0.5 * (to.log(2. * to.Tensor([to.pi]))+ to.log(self.sigma2)[:,None,:]+
                     ((x[self.marorder:])[None,:]-tmu)**2/self.sigma2[:,None,:]),dim=2)


    def compute_probt_all(self, x: to.Tensor, cuts: list[int], meant : list) -> list[to.Tensor]:
        """Computes the temporal log-likelihood for each time series 

        Args:
            x (to.Tensor): concatenated time series 
            cuts (list[int]): cuts determining the start and end of each time series eg. [0, len(x)] for a single time series
            [0,len(x_1),len(x1)+len(x2)] for two time series
            meant (list): temporal means 

        Returns:
            list[to.Tensor]: temporal log-likelihood for each input time series
        """
        return [self.compute_probt(x[cuts[i]:cuts[i+1]],meant[i]) for i in range(len(cuts)-1)]


    def collect_statistics(self,x: to.Tensor, cuts: to.Tensor) -> list:
        """Computes latent probabilities and statistics to update parameters

        Args:
            x (to.Tensor): input timeseries
            cuts (to.Tensor): cuts for the timeseries
        """
        meant_all = self.compute_mut_all(x, cuts)
        prob_list = self.compute_probt_all(x, cuts, meant_all)
        self.myforback.compute_gamma(prob_list[0])
        s_p = self.myforback.act_initial()
        s_a = self.myforback.act_transition(prob_list[0])
        s_b = self.myforback.act_weights_ashmm(x, self.graphs, self.arorders, self.marorder)
        s_v = self.myforback.act_sigma2_ashmm(x, meant_all[0], self.marorder)
        loglikelihood = self.myforback()
        for i in range(1,len(prob_list)):
            self.myforback.compute_gamma(prob_list[i])
            s_p += self.myforback.act_initial()
            s_a += self.myforback.act_transition(prob_list[i])
            s_b += self.myforback.act_weights_ashmm(x, self.graphs, self.arorders, self.marorder)
            s_v += self.myforback.act_sigma2_ashmm(x, meant_all[i], self.marorder)
            loglikelihood += self.myforback()
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
        # print("New parameters:")
        # print("*"*20)
        # print("A: ",self.transition)
        # print("*"*20)
        # print("pi: ", self.initial)
        # print("*"*20)
        # print("B: ", self.weights)
        # print("*"*20)
        # print("S: ", self.sigma2)


    def compute_EM(self, x: to.Tensor, cuts: to.Tensor):
        """Performs the EM algorithm for a fixed graph and AR-order

        Args:
            x (to.Tensor): input sequences
            cuts (to.Tensor): cuts of the time series
        """
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
