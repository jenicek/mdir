from . import cirscore

SCORES = {
    "cirdatasetap": cirscore.CirDatasetAp,
}

def initialize_score(params):
    return SCORES[params.pop("type")](params)
