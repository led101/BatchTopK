from evo2 import Evo2          # pip install git+https://github.com/ArcInstitute/evo2.git
import torch

def load_evo2(model_name="evo2_7b", dtype=torch.bfloat16):
    model = Evo2(model_name).model          # StripedHyena model object
    model.eval().to(dtype)                  # no training of Evo2 itself
    return model