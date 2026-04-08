import torch as to
from src.models.forback import ForwardBackward
from test.mocks import *

def test_forback_forwardbackward():
    subject = ForwardBackward(TRANSITION, INITIAL, NSTATES)
    subject.forward_pass(PROBT)
    subject.backward_pass(PROBT)
    subject.compute_gamma(PROBT)
    alpha = subject.alpha
    beta = subject.beta
    gamma = to.exp(subject.gamma)
    assert alpha.shape == to.Size([BATCH,NSTATES])
    assert beta.shape == to.Size([BATCH,NSTATES])
    assert gamma.shape == to.Size([BATCH,NSTATES])
    assert to.sum(gamma[0]).item() > 1- 1e-5
    counter = 0
    for i in range(BATCH):
        if to.sum(gamma[i]).item() > 1- 1e-5:
            counter+=1
    assert counter == BATCH
