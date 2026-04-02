import torch as to
from torch import nn
from src.pydant import ModelGeneralConfig
from src.models.AsHMM.BayesianNetworks import LGBayesianNetwork

class AsHMM(nn.Module):

    def __init__(self, config: ModelGeneralConfig):
        super(AsHMM, self).__init__()
        self.transition = nn.Parameter(to.ones([config.nhidden, config.nhidden])/config.nhidden)
        self.initial = nn.Parameter(to.ones(config.nhidden)/config.nhidden)
        self.graphs = nn.Parameter(to.zeros([config.nhidden, config.nfeatures, config.nfeatures]))
        self.arorders = nn.Parameter(to.zeros([config.nhidden, config.nfeatures]))
        self.sigma2 = nn.Parameter(to.ones([config.nhidden, config.nfeatures]),requires_grad=False)
        self.weights = nn.Parameter(to.zeros([config.nhidden, config.nfeatures, config.nfeatures + config.arorder]),requires_grad=False)
        self.nstates = config.nhidden
        self.marorder = config.arorder
        self.nfeatures = config.nfeatures
        self.lgnetworks = LGBayesianNetwork(self.graphs)
        self.relevancies = config.saliencies

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
        self.weights.requires_grad=True
        self.sigma2.requires_grad=True

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
        return to.sum(-0.5*(to.log(2.* to.Tensor([to.pi]))+2*to.log(self.sigma2)[:,None,:]+
                     (((x[self.marorder:])[None,:]-tmu)/self.sigma2[:,None,:])**2),dim=2)

    def compute_probt_all(self, x: to.Tensor, cuts: list[int]) -> list[to.Tensor]:
        """Computes the temporal log-likelihood for each time series 

        Args:
            x (to.Tensor): concatenated time series 
            cuts (list[int]): cuts determining the start and end of each time series eg. [0, len(x)] for a single time series
            [0,len(x_1),len(x1)+len(x2)] for two time series

        Returns:
            list[to.Tensor]: temporal log-likelihood for each input time series
        """
        meant_all = self.compute_mut_all(x, cuts)
        return [self.compute_probt(x[cuts[i]:cuts[i+1]],meant_all[i]) for i in range(len(cuts)-1)]
    
    def e_step()
    

