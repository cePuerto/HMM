import torch as to
from torch import nn
from src.models.AsHMM.BayesianNetworks import LGBayesianNetwork as lgb


class ForwardBackward(nn.Module):

    def __init__(self, transition: to.Tensor, pi: to.Tensor, nstates: int):
        super(ForwardBackward, self).__init__()
        self.transition = transition
        self.pi = pi
        self.nstates = nstates
        self.alpha = None
        self.beta = None
        self.clist = None
        self.gamma = None
        self.phi = None
        self.psi = None


    def forward_step(self, alfa: to.Tensor, probt: to.Tensor, t: int) -> to.Tensor:
        """ Does an inductive step in the alfa variable

        Args:
            alfa (to.Tensor): forward variable
            probt (to.Tensor): temporal probabilities
            t (int): time index

        Returns:
            to.Tensor: next forward variable
        """
        arg = to.exp(alfa) @ self.transition
        arg = to.clip(arg, 1e-8)
        return probt[:,t]+ to.log(arg)


    def backward_step(self,beta : to.Tensor, probt : to.Tensor, t : int)-> to.Tensor:
        """ An iteration in the backward variable

        Args:
            beta (to.Tensor): backward variable
            probt (to.Tensor): temporal probabilities
            t (int): time index

        Returns:
            to.Tensor: next backward variable
        """
        maxi = to.max(beta)
        arg = self.transition @ to.exp(probt[:,t]+beta-maxi)
        arg = to.clip(arg,1e-8)
        return  maxi+to.log(arg)


    def forward_pass(self, probt: to.Tensor):
        """Forward pass of theforward-backward algorithm

        Args:
            probt (to.Tensor): Temporal probabilities
        """
        length = len(probt[0])
        pi = to.clip(self.pi,1e-8)
        alfa = to.log(pi)+ probt[:,0]
        cd = -to.max(alfa)-to.logsumexp(alfa-to.max(alfa),0)
        Clist = to.Tensor([cd])
        alfa = alfa+cd
        Alfa = [alfa]
        for t in range(1,length):
            alfa = self.forward_step(alfa, probt, t)
            cd = -to.max(alfa)-to.logsumexp(alfa-to.max(alfa),0)
            Clist = to.cat([Clist,to.Tensor([cd])])
            alfa = cd + alfa
            Alfa.append(alfa)
        self.alpha = to.stack(Alfa)
        self.clist = Clist


    def backward_pass(self, probt: to.Tensor):
        """Backwars pass of the forward-backward algorithm

        Args:
            probt (to.Tensor): temporal probabilities
        """
        length = len(probt[0])
        nClist = self.clist.flip(dims=[0])
        beta = to.zeros([self.nstates])
        Beta = [beta]
        for t in range(1,length):
            beta = self.backward_step(beta, probt, length-t)
            beta = beta + nClist[t]
            Beta.append(beta)
        self.beta = to.flip(to.stack(Beta),dims=[0])


    def compute_gamma(self, probt: to.Tensor):
        """
        Compute Gamma or the latent probabilities
        """
        self.forward_pass(probt)
        self.backward_pass(probt)
        num = self.alpha +self.beta
        den = to.log(to.sum(to.exp(self.alpha+self.beta),dim=1))[None,:].T
        self.gamma = num-den


    def act_transition(self,probt: to.Tensor):
        """Computes statistics to update the transition matrix

        Args:
            probt (to.Tensor): temporal probabilities
        """
        num = self.transition*(to.exp(self.alpha[:-1].T) @  to.exp((self.beta[1:] + probt[:,1:].T)))
        den = to.sum(num,dim=1)[:,None]
        return [num, den]


    def act_initial(self) -> to.Tensor:
        """Computes statistics to update initial distribution parameter

        Returns:
            to.Tensor: initial numerator statistic
        """
        return to.exp(self.gamma[0])


    def compute_psiphi(self, probtk: to.Tensor, pfit: to.Tensor, pgt: to.Tensor, rho: to.Tensor):
        """Compute latent probabilities of relecant and not relevant features

        Args:
            probtk (to.Tensor): temporal probability for each feature 
            pfit (to.Tensor): temporal probability for each feature when relevant
            pgt (to.Tensor): temporal probability for each feature when not relevant
            rho (to.Tensor): relevancy parameter
        """
        self.phi = []
        self.psi = []
        for i in range(self.nstates):
            psii =  rho[i]    *to.exp(-probtk[:,i,:]+ pfit[:,i]+(self.gamma.T[i])[:,None])
            phii =  (1-rho[i])*to.exp(-probtk[:,i,:]+ pgt      +(self.gamma.T[i])[:,None])
            psii = to.clip(psii,1e-8)
            phii = to.clip(phii,1e-8)
            self.phi.append(phii)
            self.psi.append(psii)
        self.phi = to.Tensor(self.phi)
        self.psi = to.Tensor(self.psi)


    def act_rho(self) -> list: #Revisar con cuidado
        """Compute statistics to update relevancy parameter 

        Returns:
            list: [numerator, denominator] updating statistics
        """
        rho_numerator = to.sum(self.psi,dim=1)
        rho_denominator = to.sum(self.gamma,dim=0)[:,None]
        return [rho_numerator, rho_denominator]


    def act_weights_ashmm(self,x: to.Tensor, graphs: to.Tensor, arorders: to.Tensor, maxar: int) -> list: 
        """Compute the required statistics to update ashmm weigths

        Args:
            x (to.Tensor): input time series
            graphs (to.Tensor): tensor representing graphs
            arorders (to.Tensor): tensor representing AR orders
            maxar (int): maximum AR order

        Returns:
            list: [B, a] updating statistics, the parameter is the solution to Bx = a
        """
        length = len(x)
        nfeatures = len(x[0])
        bc = []
        ac = []
        for i in range(self.nstates):
            gi = graphs[i]
            wi = to.exp(self.gamma[:, i][:,None])
            arori = arorders[i]
            ack = []
            bck = []
            for k in range(nfeatures):
                pak = lgb.my_parents(gi,k)
                y = to.cat([to.ones([length - maxar, 1]), x[maxar:,pak]], axis=1)
                if arori[k] > 0:
                    z = to.stack([x[maxar - j : -j, k] for j in range(1, arori[k] + 1)]).transpose(0,1)
                    y = to.cat([y, z], dim=1)
                wiy = wi * y
                a = to.sum(wiy * x[maxar:, k][:,None],dim=0)
                b = [to.sum(wiy, dim=0)]
                for pa in pak:
                    b.append(to.sum(wiy * x[maxar:,pa][:,None] ,dim=0))
                for j in range(1, arori[k] + 1):
                    zi = x[maxar - j : -j, k][:,None]
                    b.append(to.sum(wiy * zi,dim=0))
                bck.append(to.stack(b).transpose(0,1))
                ack.append(a)
            bc.append(bck)
            ac.append(ack)
        return [bc, ac]


    def act_sigma2_ashmm(self, x: to.Tensor, mut: to.Tensor, maxar: int) -> list:
        """Computes required statistics to compute ashmm variances 

        Args:
            x (to.Tensor): input time series
            mut (to.Tensor): temporal means
            maxar (int): maximum AR order

        Returns:
            list: [numerator, denominator] updating statistics
        """
        nums = to.sum(to.exp(self.gamma.T[:,:,None])*((x[maxar:][None,:] - mut)**2),dim=1)
        dens = to.sum(to.exp(self.gamma),dim=0)[:,None]
        return [nums, dens]


    def clear_statistics(self):
        """ Clear all statistics
        """
        self.alpha = None
        self.beta = None
        self.clist = None
        self.gamma = None
        self.phi = None
        self.psi = None


    def forward(self) -> to.Tensor:
        """ Computes the log-likelihood of the time series based on the current latent statistics

        Returns:
            to.Tensor: log likelihood of the time series
        """
        return to.sum(-self.clist)
