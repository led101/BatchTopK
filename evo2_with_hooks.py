import torch

class Evo2WithHooks(torch.nn.Module):
    def __init__(self, striped_hyena):
        super().__init__()
        self.base = striped_hyena
        self.module_map = dict(self.base.named_modules())
    
    # forward every *other* attribute request to self.base
    def __getattr__(self, name):
        try:                       # avoid infinite recursion
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    # forward that mimics TL's return_type convention
    def forward(self, input_ids, *, return_type=None):
        out = self.base(input_ids) # tuple or Tensor
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        if return_type == "loss":
            # standard LM cross‑entropy
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss
        return logits

    def run_with_hooks(self, input_ids, *, fwd_hooks, return_type="loss"):
        handles = []
        for name, fn in fwd_hooks:
            mod = self.module_map[name]
            handles.append(mod.register_forward_hook(
                lambda m, inp, out, fn=fn: fn(out, m)
            ))
        try:
            return self(input_ids, return_type=return_type)
        finally:
            for h in handles:
                h.remove()