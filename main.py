#%%
from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer


for l1_coeff in [0.004, 0.0018, 0.0008]:
    cfg = get_default_cfg()
    cfg["sae_type"] = "batchtopk"
    cfg["model_name"] = "arcinstitute/evo2_7b"
    cfg["layer"] = 26
    cfg["site"] = "resid_pre"
    cfg["dataset_path"] = "arcinstitute/opengenome2"
    cfg["aux_penalty"] = (1/32)
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = True
    cfg["top_k"] = 32
    cfg['wandb_project'] = 'evo2_sae_analysis'
    cfg['device'] = 'cuda'
    cfg['bandwidth'] = 0.001
    cfg['l1_coeff'] = l1_coeff
    cfg["act_size"] = 4096  # hidden size of 7b model
    cfg["dict_size"] = 4096 * 16

    cfg = post_init_cfg(cfg)
                
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    
    if cfg["sae_type"] == "vanilla":
        sae = VanillaSAE(cfg)
    elif cfg["sae_type"] == "topk":
        sae = TopKSAE(cfg)
    elif cfg["sae_type"] == "batchtopk":
        sae = BatchTopKSAE(cfg)
    elif cfg["sae_type"] == 'jumprelu':
        sae = JumpReLUSAE(cfg)
    train_sae(sae, activations_store, model, cfg)