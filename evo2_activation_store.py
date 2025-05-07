# evo2_activation_store.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from evo2_loader import load_evo2
from datasets import load_dataset

class Evo2ActivationsStore:
    def __init__(self, cfg):
        self.cfg=cfg
        model, tokenizer = load_evo2(cfg["model_name"], cfg["dtype"])
        self.model = model
        self.device  = cfg["device"]
        self.seq_len = cfg["seq_len"]
        self.batch   = cfg["model_batch_size"]
        self.buffer_batches = cfg["num_batches_in_buffer"]
        self.tokenizer = tokenizer


        # ------------------------------------------------------------------
        # choose which module to hook
        if "evo2_hook_module_path" in cfg:
            target = self.model.get_submodule(cfg["evo2_hook_module_path"])
        else:
            # backward‑compat: fall back to integer index
            target = self.model.blocks[cfg["layer"]]
        # ------------------------------------------------------------------

        self._latest_act = None
        def hook(_, __, out):
            self._latest_act = out.detach()
        target.register_forward_hook(hook)

        ds = load_dataset(cfg["dataset_path"], split="train", streaming=True)
        self.dataset_iter = iter(ds)

    # tokenisation ---------------------------------------------------
    def _next_token_batch(self):
        # Evo 2 uses CharLevelTokenizer(512) inside model.tokenizer
        toks = []
        while len(toks) < self.batch * self.seq_len:
            ex = next(self.dataset_iter)
            seq = ex["sequence"] if "sequence" in ex else ex["text"]
            toks += self.tokenizer.tokenize(seq)
        toks = toks[: self.batch * self.seq_len]
        ids  = torch.tensor(toks, dtype=torch.long, device=self.device)
        return ids.view(self.batch, self.seq_len)

    # activation fetch ----------------------------------------------
    @torch.no_grad()
    def _get_acts(self, token_batch):
        _ = self.model(token_batch)
        acts = self._latest_act                 # (B, L, d)
        return acts.reshape(-1, acts.size(-1))  # flatten to (B·L, d)

    # public iterator ------------------------------------------------
    def next_batch(self):
        if not hasattr(self, "_buff") or len(self._buff) == 0:
            self._fill_buffer()
        return self._buff_iter.__next__()[0]

    def _fill_buffer(self):
        acts = []
        for _ in range(self.buffer_batches):
            tokens = self._next_token_batch()
            acts.append(self._get_acts(tokens))
        acts = torch.cat(acts, dim=0)
        self._buff = TensorDataset(acts)
        self._buff_iter = iter(DataLoader(self._buff,
                                          batch_size=self.cfg["batch_size"],
                                          shuffle=True))
