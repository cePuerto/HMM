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


class TrainModelConfig(BaseModel):
    nepochs : int
    learningrate: float
    epsilon: float
    inputfiles : list[str] | None
    left2right : bool | None
    checkpoint: str | None
    savepath : str | None
    viterbi : ViterbiConfig | None
    bayesnet : BNSettings | None
    relevancy : RelevancySettings | None


class TestModelConfig(BaseModel):
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
    training : TrainModelConfig | None
    testing : TestModelConfig | None


def return_schema(path: str):
    """ Export schema

    Args:
        path (src): path where the schema will be saved
    """
    main_model_schema = ModelGeneralConfig.model_json_schema()
    with open(path, 'w',encoding="utf-8") as f:
        json.dump(main_model_schema, f,ensure_ascii=False,indent=4)
