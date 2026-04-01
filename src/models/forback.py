import torch as to
from torch import nn


class ForwardBackward(nn.Module):

    def __init__(self, transition: to.Tensor, pi: to.Tensor, probt : to.Tensor):
        super(ForwardBackward, self).__init__()
        self.transtion = transition
        self.pi = pi
        nstates, length, nfeatures = probt.shape
        self.nstates = nstates
        self.length = length
        self.nfeatures= nfeatures
        self.alpha = None
        self.beta = None
        self.clist = None
        self.gamma = None
        self.phi = None
        self.psi = None
        self.pi_numerator = None
        self.tra_numerator = None
        self.tra_denominator = None
        self.rho_numerator = None
        self.rho_denominator = None


    def forward_pass(self, probt: to.Tensor):
        """Forward pass of theforward-backward algorithm

        Args:
            probt (to.Tensor): Temporal probabilities
        """
        pi = to.clip(self.pi,1e-8)
        alfa = to.log(pi)+ probt[0]
        cd = -to.max(alfa)-to.log(to.sum(to.exp(alfa-to.max(alfa))))
        Clist = to.Tensor([cd])
        alfa = alfa+cd
        Alfa = [alfa]
        for t in range(1,self.length):
            alfa = self.forward_step(alfa, probt, t)
            cd = -to.max(alfa)-to.log(to.sum(to.exp(alfa-to.max(alfa))))
            Clist = to.cat([Clist,[cd]])
            alfa = cd + alfa
            Alfa.append(alfa)
        Alfa = to.Tensor(Alfa)
        self.alpha = Alfa
        self.clist = Clist


    def backward_pass(self, probt: to.Tensor):
        """Backwars pass of the forward-backward algorithm

        Args:
            probt (to.Tensor): temporal probabilities
        """
        beta = to.zeros(self.nstates)
        nClist = self.clist[::-1]
        beta = beta + nClist[0]
        Beta = [beta]
        for t in range(1,self.length):
            beta = self.backward_step(beta, probt, self.length-t)
            beta = beta + nClist[t]
            Beta.append(beta)
        Beta= to.flipud(Beta)
        self.beta = Beta


    def compute_gamma(self):
        """
        Compute Gamma or the latent probabilities
        """
        num = self.alpha +self.beta
        den = to.log(to.sum(to.exp(self.alpha+self.beta),dim=1))[None,:].T
        self.gamma = num-den


    def forward_step(self, alfa: to.Tensor, probt: to.Tensor, t: int) -> to.Tensor:
        """ Does an inductive step in the alfa variable

        Args:
            alfa (to.Tensor): forward variable
            probt (to.Tensor): temporal probabilities
            t (int): time index

        Returns:
            to.Tensor: next forward variable
        """
        arg = to.exp(alfa) @ self.transtion
        arg = to.clip(arg, 1e-8)
        return probt[t]+ to.log(arg)


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
        arg = to.dot(self.transition,to.exp(probt[t]+beta-maxi))
        arg = to.clip(arg,1e-8)
        return  maxi+to.log(arg)


    def act_transition(self,probt: to.Tensor):
        """Computes statistics to update the transition matrix

        Args:
            probt (to.Tensor): temporal probabilities
        """
        bj = probt
        nume = []
        deno = []
        for i in range(self.nstates):
            alfat = (self.alpha.T[i])[:self.length-1] 
            betat = self.beta[1:].T 
            num = self.transtion[i]*to.sum(to.exp(alfat+betat+ bj[1:].T),dim=1)
            den = to.sum(num)
            nume.append(num)
            deno.append(den)
        self.tra_numerator = to.Tensor(nume)
        self.tra_denominator = to.Tensor(deno)


    def act_initial(self):
        """ Computes statistics to update initial distribution parameter
        """
        self.pi_numerator = to.exp(self.gamma[0])

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


    def act_rho(self): #Revisar con cuidado
        """
        Compute statistics to update relevancy parameter 
        """
        self.rho_numerator = to.sum(self.psi,dim=1)
        self.rho_denominator = to.sum(self.gamma,dim=0)[:,None]


    def forward(self, probt: to.Tensor) -> to.Tensor:
        """ Computes the log-likelihood of the time series

        Args:
            probt (to.Tensor): temporal probabilities

        Returns:
            to.Tensor: log likelihood of the time series
        """
        self.forward_pass(probt)
        self.backward_pass(probt)
        return to.sum(-self.clist)
