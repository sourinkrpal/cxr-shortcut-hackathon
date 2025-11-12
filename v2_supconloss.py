# -----------------------------------------------------------------------------
# CXP Pneumothorax — BCE + Supervised Contrastive (NO adversary)
#
# CHANGES you requested (vs last run):
#   (1) Remove BCE importance weighting (plain mean BCE now).
#       We KEEP the GroupStratifiedSampler (balanced per-batch across 4 groups),
#       but we DO NOT re-weight the BCE by true/sampler anymore.
#   (2) Make SupCon a bit stronger and softer:
#           LAMBDA_SUPCON = 0.20
#           TAU = 0.20
#
# Everything else is unchanged: same DenseNet-121 backbone, same augmentations,
# EMA for validation/testing, same logging, same test protocol.
# -----------------------------------------------------------------------------

import os
import os.path
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.models import densenet121
from torcheval.metrics import BinaryAUROC

# -------------------- Perf knobs ---------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

# -------------------- Hyperparams --------------------------------------------
LR = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 128                       # keep this as requested
NUM_WORKERS = 12

# SupCon settings (your requested change)
LAMBDA_SUPCON = 0.20                   # was 0.10 → stronger pull toward invariance
TAU = 0.20                             # was 0.10 → slightly softer temperature

# -------------------- Logging / W&B ------------------------------------------
def setup_logging(root_dir: Path):
    log_path = root_dir / "cxr_pneu.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = exception_handler


def setup_wandb(root_dir: Path):
    wandb_dir = root_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.abspath(wandb_dir)
    wandb.init(
        project="cxr_small_data_pneu",
        dir=wandb_dir,
        config={
            "learning_rate": LR,
            "architecture": "densenet121",
            "dataset": "CheXpert",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "supcon_lambda": LAMBDA_SUPCON,
            "supcon_tau": TAU,
        }
    )

# -------------------- Dataset (AUGMENTATIONS UNCHANGED) ----------------------
class CXP_dataset(torchvision.datasets.VisionDataset):
    """
    Returns (image_tensor, pneu_label (0/1), drain (0/1)).
    Augmentations are kept IDENTICAL to your previous run.
    """
    def __init__(self, root_dir, csv_file, augment=True, inference_only=False) -> None:
        if augment:
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda i: torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=20),
                transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.75, 1.3)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda i: torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        super().__init__(root_dir, transform)

        df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # keep the same CheXpert path mapping
        self.path = df.Path.str.replace('CheXpert-v1.0/', 'CheXpert-v1.0-small/', regex=False)
        self.idx = df.index
        self.transform = transform

        self.labels = df.Pneumothorax.astype(int)  # y
        self.drain  = df.Drain.astype(int)         # d

    def __getitem__(self, index: int):
        try:
            img = torchvision.io.read_image(os.path.join(self.root_dir, self.path[index]))
            img = self.transform(img)
            return img, self.labels[index], self.drain[index]
        except RuntimeError as e:
            logging.error(f"Error loading image at index {index}: {self.path[index]}")
            logging.error(f"Error message: {e}")
            # Skip problematic file and move on
            return self.__getitem__((index + 1) % len(self))

    def __len__(self) -> int:
        return len(self.path)

# -------------------- Group-stratified sampler (unchanged) -------------------
class GroupStratifiedSampler(torch.utils.data.Sampler):
    """
    Ensures each batch contains equal counts of the 4 groups:
      g = (P,D) in {(0,0),(0,1),(1,0),(1,1)}.
    Batch size MUST be divisible by 4.
    """
    def __init__(self, labels: np.ndarray, drains: np.ndarray, batch_size: int):
        super().__init__(None)
        if batch_size % 4 != 0:
            raise ValueError("Batch size must be divisible by 4 for 4 groups.")
        self.batch_size = batch_size
        self.samples_per_group = batch_size // 4
        self.num_samples = len(labels)
        self.num_batches = self.num_samples // self.batch_size

        groups = labels.astype(int) * 2 + drains.astype(int)  # 0..3
        self.group_indices = [np.where(groups == i)[0] for i in range(4)]
        self.group_pointers = [0] * 4
        for i in range(4):
            np.random.shuffle(self.group_indices[i])

    def __iter__(self):
        all_indices = []
        for _ in range(self.num_batches):
            batch_indices = []
            for i in range(4):
                start = self.group_pointers[i]
                end = start + self.samples_per_group
                if end > len(self.group_indices[i]):
                    np.random.shuffle(self.group_indices[i])
                    start = 0
                    end = self.samples_per_group
                batch_indices.extend(self.group_indices[i][start:end])
                self.group_pointers[i] = end
            np.random.shuffle(batch_indices)  # mix groups within the batch
            all_indices.extend(batch_indices)
        return iter(all_indices)

    def __len__(self):
        return self.num_batches * self.batch_size

# -------------------- Model: DenseNet + small projection head ----------------
class CXP_Model(nn.Module):
    """
    - DenseNet121(weights=IMAGENET1K_V1) as encoder.
    - clf_head: 1-dim logit for Pneumothorax (BCEWithLogits).
    - proj_head: 1000 -> 128 (normalized) for SupCon.
    """
    def __init__(self):
        super().__init__()
        self.encoder = densenet121(weights='IMAGENET1K_V1')  # returns 1000-D
        self.clf_head = nn.Linear(1000, 1)

        # 2-layer projection head (common for contrastive learning)
        self.proj_head = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def encode_1000d(self, x):
        return self.encoder(x)  # 1000-D logits from DenseNet (acts as embedding)

    def project(self, z_1000):
        z = self.proj_head(z_1000)        # [B, 128]
        z = F.normalize(z, dim=1)         # L2-normalize for cosine similarities
        return z

    def forward(self, x):
        z = self.encode_1000d(x)          # [B, 1000]
        logit = self.clf_head(z).squeeze(-1)  # [B]
        return logit

    def predict_proba(self, x):
        return torch.sigmoid(self(x))

# -------------------- SupCon loss (standard supervised contrastive) ----------
def supervised_contrastive_loss(z_norm, labels, tau: float):
    """
    Standard Supervised Contrastive loss (Khosla et al., NeurIPS'20).
    - z_norm: [B, D] L2-normalized features.
    - labels: [B] int {0,1}.
    - tau: temperature.
    For each anchor i, positives are j!=i with labels[j]==labels[i].
    """
    device = z_norm.device
    B = z_norm.size(0)

    # Cosine similarity matrix
    sim = torch.matmul(z_norm, z_norm.T)  # [B,B], in [-1,1]
    sim = sim / tau

    # Mask out self-similarity
    logits_mask = torch.ones_like(sim, dtype=torch.bool, device=device)
    logits_mask.fill_(True)
    logits_mask.fill_diagonal_(False)  # exclude i==i

    # Positive mask: same label, j != i
    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).to(device) & logits_mask  # [B,B] boolean

    # For numerical stability: subtract max per row
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Denominator: sum over all j != i
    exp_sim = torch.exp(sim) * logits_mask
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12

    # Numerator: sum over positives
    exp_sim_pos = torch.exp(sim) * pos_mask
    num = exp_sim_pos.sum(dim=1) + 1e-12

    # For anchors with no positives (rare in class-imbalanced mini-batches),
    # we safely ignore by masking them out of the mean.
    valid_anchors = (pos_mask.sum(dim=1) > 0).float()  # [B] 0/1

    # InfoNCE-style loss per anchor
    loss_i = -torch.log(num / denom).squeeze()

    # Average over valid anchors only
    loss = (loss_i * valid_anchors).sum() / (valid_anchors.sum() + 1e-12)
    return loss

# -------------------- Train/Eval ------------------------------------------------
def train_and_eval(data_dir: Path, csv_dir: Path, out_dir: Path):
    # Datasets
    train_data = CXP_dataset(data_dir, csv_dir / 'train_drain_shortcut.csv')
    val_data   = CXP_dataset(data_dir, csv_dir / 'val_drain_shortcut.csv', augment=False)
    test_data_aligned    = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_aligned.csv', augment=False)
    test_data_misaligned = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_misaligned.csv', augment=False)

    # Group-stratified batch sampler (balanced batches across 4 groups)
    labels_np = train_data.labels.to_numpy()
    drains_np = train_data.drain.to_numpy()
    sampler = GroupStratifiedSampler(labels_np, drains_np, batch_size=BATCH_SIZE)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, sampler=sampler,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )
    test_loader_aligned = torch.utils.data.DataLoader(
        test_data_aligned, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )
    test_loader_misaligned = torch.utils.data.DataLoader(
        test_data_misaligned, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )

    # Model + EMA
    model = CXP_Model().to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)

    # Losses / Optimizer / Scheduler / Metrics
    criterion_bce = nn.BCEWithLogitsLoss(reduction='mean')  # <<< CHANGE (1): plain mean BCE
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    train_auroc = BinaryAUROC()
    val_auroc   = BinaryAUROC()
    test_auroc_aligned    = BinaryAUROC()
    test_auroc_misaligned = BinaryAUROC()

    # (Optional) Log group counts for sanity (unchanged; not used in loss anymore)
    groups_np = labels_np * 2 + drains_np
    counts = np.bincount(groups_np, minlength=4).astype(float)
    logging.info(f"Group counts (true): {counts}")
    logging.info("Importance weights : [DISABLED for BCE]  (we still balance batches via the sampler)")

    best_val_loss = 1e9

    # -------------------- Training loop --------------------------------------
    for epoch in range(NUM_EPOCHS):
        logging.info(f"======= EPOCH {epoch}  (λ_con={LAMBDA_SUPCON:.2f}, τ={TAU:.2f}) =======")

        # TRAIN
        model.train()
        train_auroc.reset()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_supcon_sum = 0.0
        train_brier_sum = 0.0

        for inputs, labels, drains in tqdm(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # drains are not used by the loss anymore (no adversary, no weighting)
            optimizer.zero_grad(set_to_none=True)

            # Classification path
            logits = model(inputs)  # [B]
            bce_loss = criterion_bce(logits, labels.float())

            # Contrastive path
            with torch.no_grad():
                z_1000 = model.encode_1000d(inputs)  # [B,1000]
            z_proj = model.project(z_1000)          # [B,128] normalized
            supcon_loss = supervised_contrastive_loss(z_proj, labels, tau=TAU)

            # Total loss
            loss = bce_loss + LAMBDA_SUPCON * supcon_loss
            loss.backward()
            optimizer.step()

            # Metrics bookkeeping
            B = inputs.size(0)
            train_loss_sum += loss.item() * B
            train_bce_sum  += bce_loss.item() * B
            train_supcon_sum += supcon_loss.item() * B

            train_auroc.update(logits.detach(), labels)
            probs = torch.sigmoid(logits.detach())
            train_brier_sum += ((probs - labels.float()) ** 2).sum().item()

        # VALIDATION (EMA)
        ema_model.update_parameters(model)
        ema_model.eval()
        val_auroc.reset()
        val_loss_sum = 0.0
        val_bce_sum = 0.0
        val_supcon_sum = 0.0
        val_brier_sum = 0.0

        with torch.no_grad():
            for inputs, labels, drains in tqdm(val_loader):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                logits = ema_model(inputs).reshape(-1)  # [B]
                bce_loss = criterion_bce(logits, labels.float())

                # compute SupCon on EMA features for reporting symmetry
                z_1000 = ema_model.module.encode_1000d(inputs)  # AveragedModel exposes .module
                z_proj = ema_model.module.project(z_1000)
                supcon_loss = supervised_contrastive_loss(z_proj, labels, tau=TAU)

                total = bce_loss + LAMBDA_SUPCON * supcon_loss

                B = inputs.size(0)
                val_loss_sum += total.item() * B
                val_bce_sum  += bce_loss.item() * B
                val_supcon_sum += supcon_loss.item() * B

                val_auroc.update(logits, labels)
                probs = torch.sigmoid(logits)
                val_brier_sum += ((probs - labels.float()) ** 2).sum().item()

        # Epoch metrics
        Ntr, Nval = len(train_data), len(val_data)
        train_loss = train_loss_sum / Ntr
        val_loss   = val_loss_sum / Nval
        train_brier = train_brier_sum / Ntr
        val_brier   = val_brier_sum / Nval
        tr_auroc = train_auroc.compute().item()
        va_auroc = val_auroc.compute().item()

        logging.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} AUROC: {tr_auroc:.4f} Brier: {train_brier:.4f}\n"
            f"             Val   Loss: {val_loss:.4f} AUROC: {va_auroc:.4f} Brier: {val_brier:.4f}"
        )

        wandb.log({
            "Loss/train": train_loss,
            "Loss/val": val_loss,
            "Loss/train_bce": train_bce_sum / Ntr,
            "Loss/train_supcon": train_supcon_sum / Ntr,
            "Loss/val_bce": val_bce_sum / Nval,
            "Loss/val_supcon": val_supcon_sum / Nval,
            "auroc/train": tr_auroc,
            "auroc/val": va_auroc,
            "brier/train": train_brier,
            "brier/val": val_brier,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auroc = va_auroc
            logging.info(f"Saving new best chkpt at epoch {epoch}.")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, out_dir / 'cxp_pneu_densenet.chkpt')

        scheduler.step()

    wandb.finish()

    # -------------------- TESTING (EMA checkpoint) ---------------------------
    model = CXP_Model().to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    checkpoint = torch.load(out_dir / 'cxp_pneu_densenet.chkpt', map_location=device)
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    ema_model.eval()
    logging.info(f"Best val AUROC (from training): {best_val_auroc:.4f}")

    criterion_eval = nn.BCEWithLogitsLoss(reduction='mean')

    # Aligned
    test_loss_aligned = 0.0
    test_auroc_aligned.reset()
    test_brier_sum = 0.0
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_aligned):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = ema_model(inputs).reshape(-1)
            loss = criterion_eval(outputs, labels.float())
            test_loss_aligned += loss.item() * inputs.size(0)
            test_auroc_aligned.update(outputs, labels)
            probs = torch.sigmoid(outputs)
            test_brier_sum += ((probs - labels.float()) ** 2).sum().item()
    test_loss_aligned /= len(test_data_aligned)
    test_brier_aligned = test_brier_sum / len(test_data_aligned)
    logging.info(f"Test Loss ALIGNED: {test_loss_aligned:.4f} AUROC: {test_auroc_aligned.compute():.4f} Brier: {test_brier_aligned:.4f}")

    # Misaligned
    test_loss_misaligned = 0.0
    test_auroc_misaligned.reset()
    test_brier_sum = 0.0
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_misaligned):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = ema_model(inputs).reshape(-1)
            loss = criterion_eval(outputs, labels.float())
            test_loss_misaligned += loss.item() * inputs.size(0)
            test_auroc_misaligned.update(outputs, labels)
            probs = torch.sigmoid(outputs)
            test_brier_sum += ((probs - labels.float()) ** 2).sum().item()
    test_loss_misaligned /= len(test_data_misaligned)
    test_brier_misaligned = test_brier_sum / len(test_data_misaligned)
    logging.info(f"Test Loss MISALIGNED: {test_loss_misaligned:.4f} AUROC: {test_auroc_misaligned.compute():.4f} Brier: {test_brier_misaligned:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Directory above /CheXpert-v1.0-small',
                        default='/data')
    parser.add_argument('--csv_dir', type=str, required=False,
                        help='Directory that contains train_drain_shortcut.csv, etc.',
                        default='.')
    parser.add_argument('--out_dir', type=str, required=False,
                        help='Directory where outputs (logs, checkpoints, plots, etc.) will be placed',
                        default='~/cxp_shortcut_out')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir).expanduser()

    setup_logging(out_dir)
    setup_wandb(out_dir)

    if torch.cuda.is_available():
        logging.info("Using GPU")
    else:
        logging.info("Using CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_and_eval(data_dir, csv_dir, out_dir)
