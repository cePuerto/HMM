import json
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
    nfeatures: int
    ncomponents: int
    discrete : bool
    saliencies: bool
    arorder : int = 0
    bayesnetwork: bool
    training : TrainModel | None
    testing : TestModel | None


def return_schema(path: str):
    """ Export schema

    Args:
        path (src): path where the schema will be saved
    """
    main_model_schema = ModelGeneralConfig.model_json_schema()
    with open(path, 'w',encoding="utf-8") as f:
        json.dump(main_model_schema, f,ensure_ascii=False,indent=4)
