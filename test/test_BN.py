import torch as to
from src.models.AsHMM.BayesianNetworks import BayesianNetwork
from test.mocks import *


def test_khantopology():
    """Test the kahn topological sorting algorithm"""
    bas = BayesianNetwork()
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
    bas = BayesianNetwork()

    mu = bas.linear_gauss_means(DATAMOCK, BN1, BN1_W, AR1, MAXAR)
    mu2 = bas.linear_gauss_means(DATAMOCK, BN2, BN2_W, AR2, MAXAR)

    assert mu.shape == to.Size([int(BATCH - MAXAR), 5])
    assert mu2.shape == to.Size([int(BATCH - MAXAR), 5])
