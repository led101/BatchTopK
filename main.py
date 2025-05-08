#%%
from training import train_sae
from sae import BatchTopKSAE
from config import get_default_cfg, post_init_cfg
from evo2_activation_store import Evo2ActivationsStore
from evo2_loader import load_evo2
import torch
from evo2_with_hooks import Evo2WithHooks 


for l1_coeff in [0.0018]:
    cfg = get_default_cfg()
    cfg["model_name"] = "evo2_7b"
    cfg["layer"] = 26
    cfg["site"] = "resid_pre"
    cfg["evo2_hook_module_path"] = "blocks.25.post_norm"
    cfg["dataset_path"] = "arcinstitute/opengenome2"
    cfg["act_size"] = 4096  # hidden size of 7b model
    cfg["dict_size"] = 4096 * 16
    cfg["top_k"] = 32
    cfg['l1_coeff'] = l1_coeff
    cfg['seq_len'] = 256
    cfg['model_batch_size'] = 128
    cfg['dtype'] = torch.bfloat16
    cfg['device'] = 'cuda'
    cfg["num_tokens"]= int(2e7) # 20 million tokens

    cfg["aux_penalty"] = (1/32)
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = True
    cfg['wandb_project'] = 'evo2_sae_analysis'
    cfg['device'] = 'cuda'
    cfg['bandwidth'] = 0.001

    cfg = post_init_cfg(cfg)
    evo2_model_raw, evo2_tokenizer = load_evo2(cfg["model_name"], cfg["dtype"])
    evo2_model = Evo2WithHooks(evo2_model_raw)

    sae  = BatchTopKSAE(cfg).to(cfg["device"])
    acts = Evo2ActivationsStore(cfg, evo2_model, evo2_tokenizer)             
    train_sae(sae, acts, evo2_model, cfg)              