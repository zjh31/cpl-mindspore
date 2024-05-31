from .trans_vg import TransVG
from .language_model.torch_bert import build_bert
def build_model(args):
    return TransVG(args)
def bulid_BERT(args):
    return build_bert(args)