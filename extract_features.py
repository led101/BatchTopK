import argparse
import wandb, torch, json, re, numpy as np, pandas as pd
from evo2_loader import load_evo2
from evo2_with_hooks import Evo2WithHooks
from sae import BatchTopKSAE
from Bio import SeqIO
import os
import tempfile

"""
Downloads a WANDB artifact (sae.pt + config.json), loads Evo 2 + SAE,
and prints the 5 most CDS-enriched features on the supplied genome.
"""
parser = argparse.ArgumentParser()
parser.add_argument("wandb_artifact", help="W&B artifact path (e.g. 'user/project/artifact:version')")
parser.add_argument("gff_path", help="Path to GFF annotation file")
parser.add_argument("fasta_path", help="Path to FASTA genome file")
args = parser.parse_args()

wandb_artifact = args.wandb_artifact
gff_path = args.gff_path 
fasta_path = args.fasta_path

device = "cuda"

# 1. ── pull checkpoint from W-&-B ─────────────────────────────────────────
api = wandb.Api()
art = api.artifact(wandb_artifact)
workdir = tempfile.mkdtemp()
art_dir = art.download(root=workdir)
cfg = json.load(open(os.path.join(art_dir, "config.json")))
sae_path = os.path.join(art_dir, "sae.pt")

raw, tok = load_evo2(cfg["model_name"], torch.bfloat16)
model = Evo2WithHooks(raw).eval().to(device)
hook_name = cfg["hook_point"]

# 3. ── build SAE module & load weights ───────────────────────────────────
sae = BatchTopKSAE(cfg).to(device)
sae.load_state_dict(torch.load(sae_path, map_location=device))
sae.eval()

# 4. ── read genome & annotations ─────────────────────────────────────────
def read_fasta(path):
    rec = SeqIO.read(path, "fasta")
    return str(rec.seq).upper()

genome = read_fasta(fasta_path)

gff_cols = ["seqid","source","type","start","end","score","strand","phase","attr"]
gff = pd.read_csv(gff_path, sep="\t", comment="#", names=gff_cols)
cds_ranges = gff[gff.type=="CDS"][["start","end"]].astype(int).values

WINDOW = 1024
STEP   = 1024

def is_cds(start,end):
    s,e = start,end
    return np.any((cds_ranges[:,0]<=e) & (cds_ranges[:,1]>=s))

windows, labels = [], []
for i in range(0, len(genome)-WINDOW, STEP):
    s, e = i, i+WINDOW
    windows.append(genome[s:e])
    labels.append(is_cds(s,e))
labels = np.array(labels)

# 5. ── helper to get SAE feature activations for a batch ─────────────────
dmodel = cfg["act_size"]
def sae_batch(seq_batch):
    # encode → token ids
    toks = tok.batch_encode_plus(seq_batch, return_tensors="pt",
                                    padding="longest", truncation=True)["input_ids"].to(device)
    # capture activations
    acts = {}
    def save(_,__,out): acts["x"]=out[0] if isinstance(out,tuple) else out
    handle = dict(model.named_modules())[hook_name].register_forward_hook(save)
    with torch.no_grad():
        model(toks)
    handle.remove()
    x = acts["x"].reshape(-1, dmodel)                     # (B·L, 4096)
    sae_out = sae(x)
    feats = sae_out["feature_acts"].view(toks.size(0), toks.size(1), -1).amax(1)  # max-pool
    return feats.cpu()

# build full feature matrix in manageable chunks
X = torch.cat([sae_batch(windows[i:i+64])
                for i in range(0, len(windows), 64)], 0)   # (N, dict_size)

# 6. ── compute enrichment score  P(fires|CDS) − P(fires|nonCDS) ──────────
fires = (X > 0).float()
pos_rate = fires[labels==1].mean(0)
neg_rate = fires[labels==0].mean(0)
enrich   = (pos_rate - neg_rate).numpy()

topk = np.argsort(enrich)[-5:][::-1]
print("\nTop-5 CDS-enriched SAE features")
print("--------------------------------")
for rank, f in enumerate(topk, 1):
    print(f"{rank:2d}. feature {f:>6d}   enrichment = {enrich[f]:.3f}   "
            f"fires in CDS: {pos_rate[f]:.3f}  fires elsewhere: {neg_rate[f]:.3f}")

# (Optional) write a BED file under /data for IGV viewing later
bed_path = f"/data/feature_{topk[0]}_cds.bed"
with open(bed_path, "w") as bed:
    for i, on in enumerate(fires[:, topk[0]]):
        if on:
            bed.write(f"{os.path.basename(fasta_path)}\t{i*STEP}\t{i*STEP+WINDOW}\n")
print(f"\nBED track for feature {topk[0]} written to {bed_path}")