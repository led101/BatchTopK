import argparse, os, tempfile, gzip, json, requests
import wandb, torch, numpy as np, pandas as pd
from io import StringIO, BytesIO
from Bio import SeqIO

from evo2_loader import load_evo2
from evo2_with_hooks import Evo2WithHooks
from sae import BatchTopKSAE


def read_fasta(path: str) -> str:
    "Return upper-case genome string from local or http(s) FASTA (optionally .gz)."
    if path.startswith(("http://", "https://")):
        r = requests.get(path, timeout=30)
        r.raise_for_status()
        handle = StringIO(gzip.decompress(r.content).decode()) if path.endswith(".gz") \
             else StringIO(r.text)
    else:
        handle = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    return str(SeqIO.read(handle, "fasta").seq).upper()


def load_gff(path: str) -> pd.DataFrame:
    cols = ["seqid","source","type","start","end","score","strand","phase","attr"]
    if path.startswith(("http://", "https://")):
        r = requests.get(path, timeout=30)
        r.raise_for_status()
        txt = gzip.decompress(r.content).decode() if path.endswith(".gz") else r.text
        return pd.read_csv(StringIO(txt), sep="\t", comment="#", names=cols)
    return pd.read_csv(path, sep="\t", comment="#", names=cols,
                       compression="gzip" if path.endswith(".gz") else None)


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive feature search (CDS vs background)")
    p.add_argument("--wandb_artifact", required=True,
                   help="user/project/artifact:version")
    p.add_argument("--gff_path", required=True)
    p.add_argument("--fasta_path", required=True)
    p.add_argument("--batch", type=int, default=16,
                   help="windows per GPU batch (reduce if OOM)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1 ── pull SAE checkpoint from W&B
    api = wandb.Api()
    art = api.artifact(args.wandb_artifact)
    work = tempfile.mkdtemp()
    art_dir = art.download(root=work)
    cfg      = json.load(open(os.path.join(art_dir, "config.json")))
    cfg["dtype"] = torch.bfloat16  # overwrite the string in the json
    sae_path = os.path.join(art_dir, "sae.pt")

    # 2 ── load Evo 2 + SAE
    raw, tok = load_evo2(cfg["model_name"], torch.bfloat16)
    model    = Evo2WithHooks(raw).eval().to(device)
    sae      = BatchTopKSAE(cfg).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()
    hook_name = "base." + cfg["hook_point"]
    d_model   = cfg["act_size"]

    # 3 ── genome + labels
    genome = read_fasta(args.fasta_path)
    gff    = load_gff(args.gff_path)
    cds    = gff[gff.type == "CDS"][["start","end"]].astype(int).values

    W, STEP = 1024, 1024
    def is_cds(s,e): return np.any((cds[:,0] <= e) & (cds[:,1] >= s))

    windows, labels = [], []
    for i in range(0, len(genome) - W, STEP):
        s, e = i, i + W
        windows.append(genome[s:e])
        labels.append(is_cds(s, e))
    labels = np.asarray(labels)

    SEQ_LEN   = cfg["seq_len"]
    PAD_ID    = 0                      # Evo 2’s PAD is zero for CharLevelTokenizer

    def encode_batch(seq_batch):
        """
        → 2-D LongTensor of shape (B, SEQ_LEN) on the current device.
        """
        out = torch.full((len(seq_batch), SEQ_LEN), PAD_ID,
                        dtype=torch.long, device=device)
        for i, s in enumerate(seq_batch):
            ids = tok.tokenize(s)[:SEQ_LEN]   # truncate if longer
            out[i, :len(ids)] = torch.tensor(ids, device=device)
        return out

    # 4 ── helper: SAE activations for a batch of windows
    def sae_batch(seq_batch):
        toks = encode_batch(seq_batch)           # your helper from earlier

        acts = {}
        hook = dict(model.named_modules())[hook_name].register_forward_hook(
            lambda m, _in, out: acts.setdefault(
                "x", out[0] if isinstance(out, tuple) else out)
        )

        with torch.no_grad():                    # TURN OFF GRADIENT TRACKING
            model(toks)                          # Evo 2 forward
        hook.remove()

        x = acts["x"].reshape(-1, d_model)       # (B·L, 4096)

        with torch.no_grad():                    # SAME FOR THE SAE CALL
            feats = sae(x)["feature_acts"]       # no graph kept

        return feats.view(toks.size(0), toks.size(1), -1).amax(1).cpu()

    # 5 ── build full feature matrix
    XS = []
    for i in range(0, len(windows), args.batch):
        XS.append(sae_batch(windows[i:i+args.batch]))
    X = torch.cat(XS)                                 # (N, dict_size)

    # 6 ── enrichment score
    fires = (X > 0).float()
    pos, neg = fires[labels==1].mean(0), fires[labels==0].mean(0)
    enrich   = (pos - neg).numpy()
    topk = np.argsort(enrich)[-5:][::-1]

    print("\nTop-5 CDS-enriched SAE features")
    for r,f in enumerate(topk,1):
        print(f"{r:2d}. f/{f:<6d}  Δ = {enrich[f]:.3f}  P(CDS)={pos[f]:.3f}  P(bg)={neg[f]:.3f}")

    # 7 ── optional BED
    os.makedirs("/data", exist_ok=True)
    bed = f"/data/feature_{topk[0]}_cds.bed"
    with open(bed, "w") as out:
        for i,on in enumerate(fires[:, topk[0]]):
            if on: out.write(f"{os.path.basename(args.fasta_path)}\t{i*STEP}\t{i*STEP+W}\n")
    print("BED track written ➜", bed)


if __name__ == "__main__":
    main()
