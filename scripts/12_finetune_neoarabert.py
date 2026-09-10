"""
Knowledge distillation: fine-tune NeoAraBERT on the silver labels and
evaluate on human-labelled BAREC.

The student is a NeoAraBERT encoder with a linear classification head on
the CLS vector. It is trained for three epochs on the silver-labelled chunks
only, never on a human annotation, then evaluated on the 1,000-paragraph
BAREC set. Two loss variants are trained and both are reported: a
class-weighted cross-entropy and a class-balanced focal loss. The
validation set selects the best epoch of each variant.

Produces
    Table 3, both rows
    Figure 1, confusion matrix data

Inputs
    outputs/silver/{train,val,test}.csv   from 11_split_silver.py
    data/barec_5levels.parquet            human-labelled evaluation set

Outputs
    outputs/distillation/<variant>/pytorch_model.bin, head_config.json,
                                    metrics.json, barec_predictions.csv
    outputs/tables/table3_distillation.csv

Usage
    python scripts/12_finetune_neoarabert.py

A GPU is strongly recommended. With BATCH = 8 and MAX_LEN = 256 each variant
takes about an hour on a single consumer GPU. Trained checkpoints are
reused on re-runs.
"""
import json
import os
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_NAME = "U4RASD/NeoAraBERT"
SILVER_DIR = "outputs/silver"
BAREC_FILE = "data/barec_5levels.parquet"
RESULTS_DIR = "outputs/distillation"
TABLES_DIR = "outputs/tables"
VARIANTS = ["weighted_ce", "focal"]
GAMMA = 2.0
MAX_LEN = 256
BATCH = 8
EPOCHS = 3
LR = 2e-5
WARMUP = 0.1
NUM_LABELS = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_split(name):
    df = pd.read_csv(os.path.join(SILVER_DIR, f"{name}.csv"))
    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def extract_cls(out):
    if hasattr(out, "last_hidden_state"):
        h = out.last_hidden_state
    elif isinstance(out, (tuple, list)):
        h = out[0]
    else:
        h = out
    return h[:, 0, :]


class NeoClassifier(nn.Module):
    def __init__(self, model_name, num_labels, hidden):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.dropout(extract_cls(out)))


class FocalLoss(nn.Module):
    """Class-balanced focal loss: alpha_c * (1 - p_t)^gamma * (-log p_t)."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = self.alpha.gather(0, target) * loss
        return loss.mean()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    print("device:", device)

    tr_x, tr_y = load_split("train")
    va_x, va_y = load_split("val")
    te_x, te_y = load_split("test")
    bdf = pd.read_parquet(BAREC_FILE)
    b_x = bdf["text"].astype(str).tolist()
    b_y = bdf["difficulty_level"].astype(int).tolist()
    print(f"silver train={len(tr_x)} val={len(va_x)} test={len(te_x)}   BAREC={len(b_x)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def collate(batch):
        t = [b[0] for b in batch]
        y = [b[1] for b in batch]
        enc = tokenizer(t, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(y, dtype=torch.long)

    tr_loader = DataLoader(TextDataset(tr_x, tr_y), batch_size=BATCH, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(TextDataset(va_x, va_y), batch_size=BATCH * 2, shuffle=False, collate_fn=collate)
    te_loader = DataLoader(TextDataset(te_x, te_y), batch_size=BATCH * 2, shuffle=False, collate_fn=collate)
    b_loader = DataLoader(TextDataset(b_x, b_y), batch_size=BATCH * 2, shuffle=False, collate_fn=collate)

    cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=np.array(tr_y))
    class_weights = torch.tensor(cw, dtype=torch.float, device=device)
    print("class weights:", [f"{w:.2f}" for w in cw])

    # Hidden size of the encoder, read from a forward pass
    enc = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device).eval()
    with torch.no_grad():
        e = tokenizer(["نص قصير"], return_tensors="pt", truncation=True, max_length=8)
        o = enc(input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device))
        hidden = extract_cls(o).shape[-1]
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()
    print("hidden size:", hidden)

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        preds, gts = [], []
        for ii, am, lb in loader:
            ii, am = ii.to(device), am.to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(ii, am)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(lb.tolist())
        return (accuracy_score(gts, preds), f1_score(gts, preds, average="macro"),
                cohen_kappa_score(gts, preds, weights="quadratic"), gts, preds)

    def train_variant(variant):
        mdir = os.path.join(RESULTS_DIR, variant)
        os.makedirs(mdir, exist_ok=True)
        ckpt = os.path.join(mdir, "pytorch_model.bin")
        model = NeoClassifier(MODEL_NAME, NUM_LABELS, hidden).to(device)
        if os.path.exists(ckpt):
            print(f"[{variant}] checkpoint found, skipping training")
            model.load_state_dict(torch.load(ckpt, map_location=device))
            _, _, best_qwk, _, _ = evaluate(model, va_loader)
            return model, mdir, best_qwk
        if variant == "focal":
            criterion = FocalLoss(alpha=class_weights, gamma=GAMMA)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total = len(tr_loader) * EPOCHS
        sched = get_linear_schedule_with_warmup(optimizer, int(WARMUP * total), total)
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        best_qwk = -1
        for ep in range(1, EPOCHS + 1):
            model.train()
            running = 0.0
            for ii, am, lb in tqdm(tr_loader, desc=f"[{variant}] epoch {ep}/{EPOCHS}"):
                ii, am, lb = ii.to(device), am.to(device), lb.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device, enabled=(device == "cuda")):
                    loss = criterion(model(ii, am), lb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                running += loss.item()
            acc, f1, qwk, _, _ = evaluate(model, va_loader)
            print(f"  [{variant}] val epoch {ep}: loss={running / len(tr_loader):.3f} "
                  f"acc={acc:.3f} f1={f1:.3f} QWK={qwk:.3f}")
            if qwk > best_qwk:
                best_qwk = qwk
                torch.save(model.state_dict(), ckpt)
                tokenizer.save_pretrained(mdir)
                json.dump({"model_name": MODEL_NAME, "num_labels": NUM_LABELS, "max_len": MAX_LEN,
                           "hidden": hidden, "variant": variant},
                          open(os.path.join(mdir, "head_config.json"), "w"))
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model, mdir, best_qwk

    summary = []
    for variant in VARIANTS:
        print(f"\n===== {variant} =====")
        try:
            model, mdir, val_qwk = train_variant(variant)
            g_acc, g_f1, g_qwk, g_gts, g_preds = evaluate(model, te_loader)
            b_acc, b_f1, b_qwk, b_gts, b_preds = evaluate(model, b_loader)
            b_gts_a, b_preds_a = np.array(b_gts), np.array(b_preds)
            b_within1 = float(np.mean(np.abs(b_gts_a - b_preds_a) <= 1))
            b_mae = float(np.mean(np.abs(b_gts_a - b_preds_a)))
            print(f"[{variant}] silver test: acc={g_acc:.3f} f1={g_f1:.3f} QWK={g_qwk:.3f}")
            print(f"[{variant}] BAREC:       exact={b_acc:.3f} within-1={b_within1:.3f} "
                  f"MAE={b_mae:.3f} QWK={b_qwk:.3f}")
            pd.DataFrame({"text": b_x, "gold": b_gts, "pred": b_preds}).to_csv(
                os.path.join(mdir, "barec_predictions.csv"), index=False, encoding="utf-8")
            json.dump({"variant": variant, "val_qwk": val_qwk,
                       "silver_test": {"acc": g_acc, "f1_macro": g_f1, "qwk": g_qwk,
                                       "confusion": confusion_matrix(g_gts, g_preds, labels=list(range(NUM_LABELS))).tolist()},
                       "barec": {"exact": b_acc, "within_1": b_within1, "mae": b_mae, "f1_macro": b_f1, "qwk": b_qwk,
                                 "confusion": confusion_matrix(b_gts, b_preds, labels=list(range(NUM_LABELS))).tolist()}},
                      open(os.path.join(mdir, "metrics.json"), "w"), indent=2, ensure_ascii=False)
            summary.append({"loss": "Weighted CE" if variant == "weighted_ce" else "Focal",
                            "exact": round(b_acc, 3), "within_1": round(b_within1, 3),
                            "mae": round(b_mae, 3), "qwk": round(b_qwk, 3)})
            if device == "cuda":
                del model
                torch.cuda.empty_cache()
        except Exception as ex:
            print(f"variant {variant} failed: {ex}")
            traceback.print_exc()
            if device == "cuda":
                torch.cuda.empty_cache()

    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(os.path.join(TABLES_DIR, "table3_distillation.csv"), index=False)
        print("\n" + sdf.to_string(index=False))
        print(f"\nwritten: {TABLES_DIR}/table3_distillation.csv")


if __name__ == "__main__":
    main()
