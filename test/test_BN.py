import torch as to
from src.models.AsHMM.BayesianNetworks import LGBayesianNetwork
from test.mocks import *

def test_myparents():
    """Test the my_parents function
    """
    bas = LGBayesianNetwork(BN1[None,:])

    pa11 = bas.my_parents(BN1,0)
    pa12 = bas.my_parents(BN1,1)
    pa13 = bas.my_parents(BN1,2)
    pa14 = bas.my_parents(BN1,3)
    pa15 = bas.my_parents(BN1,4)

    pa21 = bas.my_parents(BN2,0)
    pa22 = bas.my_parents(BN2,1)
    pa23 = bas.my_parents(BN2,2)
    pa24 = bas.my_parents(BN2,3)
    pa25 = bas.my_parents(BN2,4)


    assert to.sum(pa11 -to.Tensor([1])) == 0
    assert to.sum(pa12 -to.Tensor([2])) == 0
    assert to.sum(pa13 -to.Tensor([3])) == 0
    assert to.sum(pa14 -to.Tensor([4])) == 0
    assert to.sum(pa15 -to.Tensor([])) == 0

    assert to.sum(pa21 -to.Tensor([1,2,3,4])) == 0
    assert to.sum(pa22 -to.Tensor([2,3,4])) == 0
    assert to.sum(pa23 -to.Tensor([3,4])) == 0
    assert to.sum(pa24 -to.Tensor([4])) == 0
    assert to.sum(pa25 -to.Tensor([])) == 0


def test_khantopology():
    """Test the kahn topological sorting algorithm"""
    bas = LGBayesianNetwork(BN1[None,:])
    salida1 = bas.dag_v(BN1)
    salida2 = bas.dag_v(BN2)
    salida3 = bas.dag_v(NOTABN1)
    salida4 = bas.dag_v(NOTABN2)

    assert salida1[0] is True
    assert salida2[0] is True
    assert salida3[0] is False
    assert salida4[0] is False

    assert to.sum(salida1[1] == to.Tensor([4, 3, 2, 1, 0])) == 5
    assert to.sum(salida2[1] == to.Tensor([4, 3, 2, 1, 0])) == 5


def test_lgmean():
    """Test Bayesian mean computation"""
    bas = LGBayesianNetwork(BN1[None,:])

    mu = bas.lg_temp_mu(DATAMOCK, BN1, BN1_W, AR1, MAXAR)
    mu2 = bas.lg_temp_mu(DATAMOCK, BN2, BN2_W, AR2, MAXAR)

    assert mu.shape == to.Size([int(BATCH - MAXAR), 5])
    assert mu2.shape == to.Size([int(BATCH - MAXAR), 5])


def test_nvm():
    """Test the nvm representation computation from a LGBN
    """
    bas = LGBayesianNetwork(BN3[None,:])
    salida = bas.mvn_param(BN3,to.Tensor([0,1,2]).int(),BN3_W,SIGMA3_2,to.Tensor([0,0,0]))
    mu , sigma = salida
    expected_mu =  to.Tensor([1, -3, 4])
    expected_sigma = to.Tensor([[4, 2, 0],[2, 5, -5],[0, -5, 8]]) # This is not quite real, x3 and x1 are actually related.
    print(sigma)
    assert to.sum(expected_mu- mu) == 0
    assert to.sum(sigma -expected_sigma) == 0
