from evo2 import Evo2          # pip install git+https://github.com/ArcInstitute/evo2.git
import torch

def load_evo2(model_name="evo2_7b", dtype=torch.bfloat16):
    wrapper = Evo2(model_name)          # StripedHyena model object
    model = wrapper.model
    model.eval().to(dtype)                  # no training of Evo2 itself
    tokenizer = wrapper.tokenizer
    return model, tokenizer