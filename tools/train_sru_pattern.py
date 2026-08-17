"""Minimal learning demo: SRU must remember the previous timestep.

Task: at every timestep t >= 1, predict the binary input of timestep t - 1.
Inputs are independent random bits, so x_t carries no information about the
target x_{t-1} -- the model can only succeed through recurrent memory.

Run from the repository root:
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /ssd4/envs/prefusion_contrib_pt271cu128_py310/bin/python tools/train_sru_pattern.py
"""

import torch
from torch import nn

from robonav.aqua.model.sru import SRU

SEED = 0
NUM_SEQUENCES = 512  # split deterministically into train/validation
NUM_VAL = 128
SEQ_LEN = 8
FEATURES = 4
HIDDEN = 16
BATCH = 64
EPOCHS = 25
LR = 1e-2


def make_data(g):
    seqs = (torch.rand(NUM_SEQUENCES, SEQ_LEN, FEATURES, generator=g) < 0.5).float()
    return seqs[: NUM_SEQUENCES - NUM_VAL], seqs[NUM_SEQUENCES - NUM_VAL :]


class PrevStepPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sru = SRU(FEATURES, HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN, FEATURES)

    def forward(self, x):
        out, _ = self.sru(x)
        return self.head(out)  # logits, (batch, seq, features)


def loss_acc(model, x):
    logits = model(x)
    # Target is the previous input; timestep 0 has no previous input.
    loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 1:], x[:, :-1])
    with torch.no_grad():
        acc = (logits[:, 1:] > 0).eq(x[:, :-1]).float().mean().item()
    return loss, acc


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    train_x, val_x = make_data(g)

    model = PrevStepPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    init_loss, init_acc = loss_acc(model, val_x)
    print(f"initial   val loss {init_loss:.4f}  acc {init_acc:.3f}")

    for _ in range(EPOCHS):
        perm = torch.randperm(len(train_x), generator=g)
        for idx in perm.split(BATCH):
            loss, _ = loss_acc(model, train_x[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    train_loss, _ = loss_acc(model, train_x)
    val_loss, val_acc = loss_acc(model, val_x)
    print(f"final     train loss {train_loss:.4f}")
    print(f"final     val loss {val_loss:.4f}  acc {val_acc:.3f}")

    probs = torch.sigmoid(model(val_x[:1]))[0, 1:]
    print("example t=1..7 target vs prediction (first val sequence, bit-wise):")
    for t, (tgt, prd) in enumerate(zip(val_x[0, :-1], (probs > 0.5).float()), start=1):
        print(f"  t={t}  target {tgt.tolist()}  pred {prd.tolist()}")

    assert val_loss < init_loss, "validation loss did not decrease"
    assert val_acc >= 0.99, f"validation accuracy {val_acc:.3f} below 0.99"
    print("OK: loss decreased and validation accuracy reached the threshold")


if __name__ == "__main__":
    main()
