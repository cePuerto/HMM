import torch as to
from src.models.AsHMM.BayesianNetworks import dag_v
from test.mocks import BN1, BN2, NOTABN1, NOTABN2


def test_khantopology():
    """Test the kahn topological sorting algorithm"""
    salida1 = dag_v(BN1)
    salida2 = dag_v(BN2)
    salida3 = dag_v(NOTABN1)
    salida4 = dag_v(NOTABN2)

    assert salida1[0] is True
    assert salida2[0] is True
    assert salida3[0] is False
    assert salida4[0] is False

    assert to.sum(salida1[1] == to.Tensor([4, 3, 2, 1, 0])) == 5
    assert to.sum(salida2[1] == to.Tensor([4, 3, 2, 1, 0])) == 5
