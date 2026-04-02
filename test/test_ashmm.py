from src.models.AsHMM.model import AsHMM
from src.pydant import ModelGeneralConfig
from test.mocks import *

def test_ashmm():
    config = ModelGeneralConfig(
        nhidden=int(NSTATES),
        nfeatures=int(5),
        ncomponents=int(0),
        discrete=False,
        saliencies=False,
        arorder=int(3),
        bayesnetwork=True,
        training=None,
        testing=None)
    subject = AsHMM(config)
    subject.init_params(DATAMOCK)
    means_list = subject.compute_mut_all(DATAMOCK,[0,BATCH])
    probt = subject.compute_probt_all(DATAMOCK,[0,BATCH])
    assert means_list[0].shape == to.Size([NSTATES,BATCH-3,5])
    assert probt[0].shape == to.Size([NSTATES,BATCH-3])
    