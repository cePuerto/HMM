from pydantic import BaseModel


class ViterbiConfig(BaseModel):
    maxreduce : bool = True
    abswise : bool = False
    checkpoint: str = ""
    savepath: str = ""


class RelevancySettings(BaseModel):
    rhozero : float = 0.9
    fixed : list[int] | None = None
    nfixed : list[int] | None = None
    checkpoint: str = ""
    savepath: str = ""


class BNSettings(BaseModel):
    checkpointbn : str = ""
    savepathbn: str = ""
    checkpointtop : str = ""
    savepathtop : str = ""


class TrainModel(BaseModel):
    nepochs : int
    learningrate: float
    epsilon: float
    inputfiles : list[str]
    left2right : bool
    checkpoint: str
    savepath : str
    viterbi : ViterbiConfig
    bayesnet : BNSettings
    relevancy : RelevancySettings


class TestModel(BaseModel):
    checkpoint : str
    inputfiles : list[str]
    viterbi : ViterbiConfig
    bayesnet : BNSettings
    relevancy : RelevancySettings


class ModelGeneralConfig(BaseModel):
    nhidden: int
    ncomponents: int
    discrete : bool
    saliencies: bool
    arorder : int = 0
    bayesnetwork: bool
    training : TrainModel | None
    testing : TestModel | None
