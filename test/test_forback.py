import torch as to
from src.models.forback import ForwardBackward
from test.mocks import *

def test_forwardbackward():
    subject = ForwardBackward(TRANSITION, INITIAL, NSTATES)
    subject.compute_gamma(PROBT)
    print(subject.alpha.shape)
    alpha = subject.alpha
    beta = subject.beta
    gamma = to.exp(subject.gamma)
    assert alpha.shape == to.Size([BATCH-MAR,NSTATES])
    assert beta.shape == to.Size([BATCH-MAR,NSTATES])
    assert gamma.shape == to.Size([BATCH-MAR,NSTATES])
    assert to.sum(gamma[0]).item() > 1- 1e-5
    counter = 0
    for i in range(BATCH-MAR):
        prueba =to.sum(gamma[i]).item()
        if prueba > 1- 1e-5 and prueba < 1 +1e-5  :
            counter+=1
    assert counter == BATCH-MAR

def test_compute_sa():
    subject = ForwardBackward(TRANSITION, INITIAL, NSTATES)
    subject.compute_gamma(PROBT)
    sa = subject.act_transition(PROBT)
    num = sa[0]
    den = sa[1]
    assert num.shape == to.Size([NSTATES,NSTATES])
    assert den.shape == to.Size([NSTATES])


def test_compute_sp():
    subject = ForwardBackward(TRANSITION, INITIAL, NSTATES)
    subject.compute_gamma(PROBT)
    sp = subject.act_initial()
    assert sp.shape == to.Size([NSTATES])
    assert to.sum(sp).item() >= 1 - 1e-5
    assert to.sum(sp).item() <= 1 + 1e-5


def test_computes_sb():
    subject = ForwardBackward(TRANSITION, INITIAL, NSTATES)
    subject.compute_gamma(PROBT)
    sb = subject.act_weights_ashmm(DATAMOCK, GRAPHS, ARORDERS, MAR)
    bmatrix = sb[0]
    avector = sb[1]
    assert len(bmatrix) == NSTATES
    assert len(avector) == NSTATES
    counter_matrix = 0
    for _, state in enumerate(bmatrix):
        for j, variable in enumerate(state):
            if variable.shape == to.Size([4,4]) and j !=4:
                counter_matrix +=1
            elif variable.shape == to.Size([3,3]) and j==4:
                counter_matrix +=1
    counter_vector = 0
    for _, state in enumerate(avector):
        for j, variable in enumerate(state):
            if variable.shape == to.Size([4]) and j !=4:
                counter_vector +=1
            elif variable.shape == to.Size([3]) and j==4:
                counter_vector +=1
    assert counter_vector == NSTATES*5
    assert counter_matrix == NSTATES*5



def test_compute_sv():
    return 0
