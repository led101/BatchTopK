#%%
from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from evo2_activation_store import Evo2ActivationsStore   # <‑‑ new
from transformer_lens import HookedTransformer
import torch


for l1_coeff in [0.004, 0.0018, 0.0008]:
    cfg = get_default_cfg()
    cfg["model_name"] = "evo2_7b"
    cfg["layer"] = 26
    cfg["site"] = "resid_pre"
    evo2_hook_module_path = "blocks.25.post_norm",
    cfg["dataset_path"] = "arcinstitute/opengenome2"
    cfg["act_size"] = 4096  # hidden size of 7b model
    cfg["dict_size"] = 4096 * 16
    cfg["top_k"] = 32
    cfg['l1_coeff'] = l1_coeff
    cfg['seq_len'] = 512,
    cfg['model_batch_size'] = 256,
    cfg['dtype'] = torch.bfloat16,
    cfg['device'] = 'cuda',

    cfg["aux_penalty"] = (1/32)
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = True
    cfg['wandb_project'] = 'evo2_sae_analysis'
    cfg['device'] = 'cuda'
    cfg['bandwidth'] = 0.001

    cfg = post_init_cfg(cfg)

    sae  = BatchTopKSAE(cfg).to(cfg["device"])
    acts = Evo2ActivationsStore(cfg)             # <‑‑ new
    train_sae(sae, acts, None, cfg)              # model‑arg can be None
                
    # model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    # activations_store = ActivationsStore(model, cfg)