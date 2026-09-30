# This is cxp_pneu with the following changes:
# - multiple runs, showing mean +/- std results
# - uses balanced val set
# - optional train set balancing

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
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.models import densenet121
from torcheval.metrics import BinaryAUROC

from supcon import SupervisedContrastiveLoss

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

LR = 0.0001
NUM_EPOCHS = 30
NUM_RUNS = 10
LAMBDA_SUPCON = 0.50
TAU = 0.10

def setup_logging(root_dir):
    log_path = root_dir / "cxr_pneu.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    # Capture uncaught exceptions to log file
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_handler    


def setup_wandb(root_dir, balance_train, balance_val, select_chkpt_on, supcon=False):

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
        "balance_train": balance_train,
        "balance_val": balance_val,
        "select_chkpt_on": select_chkpt_on,
        "supcon_lambda": LAMBDA_SUPCON,
        "supcon_tau": TAU,        
        "loss": "BCE" if not supcon else "BCE+supcon"
        }
    )       


class CXP_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = densenet121(weights='IMAGENET1K_V1')
        
        # Remove the original classifier
        num_features = self.encoder.classifier.in_features
        assert num_features == 1024
        self.encoder.classifier = nn.Identity()  # Remove the classification head
        
        # using BCEWithLogitsLoss, so no sigmoid needed - but need explicit sigmoid for prob prediction
        self.clf = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Projection head (for contrastive learning)
        # 2-layer MLP as commonly used in contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        z = self.encode(x)
        return self.clf(z), self.projection_head(z)  # projection is for SupCon loss
    
    def encode(self, x):
        return self.encoder(x)
    
    def predict_proba(self, x):
        return torch.sigmoid(self(x))


class CXP_dataset(torchvision.datasets.VisionDataset):

    def __init__(self, root_dir, csv_file, augment=True, inference_only=False) -> None:

        if augment:
            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda i: torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i),
                transforms.Normalize(  # params for pretrained resnet, see https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.DenseNet121_Weights
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=20),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.75, 1.3))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda i: torch.cat([i, i, i], dim=0) if i.shape[0] == 1 else i),
                transforms.Normalize(  # params for pretrained resnet, see https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.DenseNet121_Weights
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        super().__init__(root_dir, transform)

        df = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.path = df.Path.str.replace('CheXpert-v1.0/', 'CheXpert-v1.0-small/', regex=False)
        self.idx = df.index
        self.transform = transform

        self.labels = df.Pneumothorax.astype(int)
        self.drain = df.Drain.astype(int)

    def __getitem__(self, index: int):
        try:
            img = torchvision.io.read_image(os.path.join(self.root_dir, self.path[index]))
            img = self.transform(img)
            return img, self.labels[index], self.drain[index]
        except RuntimeError as e:
            logging.error(f"Error loading image at index {index}: {self.path[index]}")
            logging.error(f"Error message: {e}")
            # Return the next valid image
            return self.__getitem__((index + 1) % len(self))        
    
    def __len__(self) -> int:
        return len(self.path)

def train_and_eval(data_dir, csv_dir, out_dir, 
                   balance_train_set=False,
                   balance_val_set=False, 
                   select_chkpt_on="loss",
                   supcon=False):
    if balance_val_set:
        train_data = CXP_dataset(data_dir, csv_dir / 'train_drain_shortcut_v2.csv')
        val_data = CXP_dataset(data_dir, csv_dir / 'val_drain_shortcut_v2.csv', augment=False)
    else:
        train_data = CXP_dataset(data_dir, csv_dir / 'train_drain_shortcut.csv')
        val_data = CXP_dataset(data_dir, csv_dir / 'val_drain_shortcut.csv', augment=False)

    test_data_aligned = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_aligned.csv', augment=False)
    test_data_misaligned = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_misaligned.csv', augment=False)
    
    if balance_train_set:
        pneu_msk = train_data.labels==1
        drain_counts_pneu = torch.bincount(torch.from_numpy(train_data.drain[pneu_msk].values))
        drain_weights_pneu = 1.0 / drain_counts_pneu.float()
        drain_counts_nopneu = torch.bincount(torch.from_numpy(train_data.drain[~pneu_msk].values))
        drain_weights_nopneu = 1.0 / drain_counts_nopneu.float()        
        sample_weights = torch.zeros_like(torch.from_numpy(train_data.labels.values), dtype=torch.float32)
        sample_weights[pneu_msk] = drain_weights_pneu[train_data.drain[pneu_msk].values]
        sample_weights[~pneu_msk] = drain_weights_nopneu[train_data.drain[~pneu_msk].values]

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True if not balance_train_set else False, num_workers=12, pin_memory=True, prefetch_factor=2,
                                               sampler=None if not balance_train_set else sampler)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)
    test_loader_aligned = torch.utils.data.DataLoader(test_data_aligned, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)      
    test_loader_misaligned = torch.utils.data.DataLoader(test_data_misaligned, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)      
    
    model = CXP_Model()
    model = model.to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    
    BCE = nn.BCEWithLogitsLoss()
    supconloss = SupervisedContrastiveLoss(temperature=TAU)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)

    train_auroc = BinaryAUROC()
    val_auroc = BinaryAUROC()
    test_auroc_aligned = BinaryAUROC()
    test_auroc_misaligned = BinaryAUROC()

    best_val_loss = 10000.0
    best_val_auroc = 0.0

    # Train the model
    for epoch in range(NUM_EPOCHS):

        print(f'======= EPOCH {epoch} =======')

        # Train
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_supcon = 0.0
        train_auroc.reset()
        train_brier_sum = 0.0
        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, projections = model(inputs)
            logits = logits.reshape(-1)

            bce_loss = BCE(logits, labels.to(torch.float32))
            supcon_loss = supconloss(projections, labels)

            if supcon:
                loss = bce_loss + LAMBDA_SUPCON * supcon_loss
            else:
                loss = bce_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_bce += bce_loss.item() * inputs.size(0)
            train_supcon += supcon_loss.item() * inputs.size(0)
            train_auroc.update(logits, labels)
            # Compute Brier Score: mean((probs - labels)^2)
            probs = torch.sigmoid(logits)
            brier = ((probs - labels.float()) ** 2).sum().item()
            train_brier_sum += brier            

        # Validation
        #model.eval()
        ema_model.update_parameters(model)
        ema_model.eval()
        
        val_loss = 0.0
        val_bce = 0.0
        val_supcon = 0.0
        val_auroc.reset()
        val_brier_sum = 0.0
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                logits, projections = ema_model(inputs)
                logits = logits.reshape(-1)
                bce_loss = BCE(logits, labels.to(torch.float32))
                supcon_loss = supconloss(projections, labels)

                if supcon:
                    loss = bce_loss + LAMBDA_SUPCON * supcon_loss
                else:
                    loss = bce_loss

                val_loss += loss.item() * inputs.size(0)
                val_bce += bce_loss.item() * inputs.size(0)
                val_supcon += supcon_loss.item() * inputs.size(0)                
                val_auroc.update(logits, labels)
                # Compute Brier Score: mean((probs - labels)^2)
                probs = torch.sigmoid(logits)
                brier = ((probs - labels.float()) ** 2).sum().item()
                val_brier_sum += brier                  

        # Print training and val performance
        train_loss /= len(train_data)
        train_bce /= len(train_data)
        train_supcon /= len(train_data)
        val_loss /= len(val_data)
        val_bce /= len(val_data)
        val_supcon /= len(val_data)
        train_brier = train_brier_sum / len(train_data)
        val_brier = val_brier_sum / len(val_data)
        logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} AUROC: {train_auroc.compute():.4f} Brier: {train_brier:.4f}\n"
              f"             Val  Loss: {val_loss:.4f} AUROC: {val_auroc.compute():.4f} Brier: {val_brier:.4f}")

        wandb.log({"Loss/train": train_loss,
                   "Loss/val": val_loss,
                   "auroc/train": train_auroc.compute(),
                   "auroc/val": val_auroc.compute(),
                   "brier/train": train_brier,
                   "brier/val": val_brier,
                   "BCE/train": train_bce,
                   "BCE/val": val_bce,
                   "SupCon/train": train_supcon,
                   "SupCon/val": val_supcon})
        
        if (select_chkpt_on.upper() == "AUROC" and val_auroc.compute() > best_val_auroc) or \
            (select_chkpt_on.upper() == "LOSS" and val_loss < best_val_loss):
            best_val_auroc = val_auroc.compute()
            best_val_loss = val_loss
            logging.info(f"Saving new best chkpt at epoch {epoch}.")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, out_dir / 'cxp_pneu_densenet.chkpt')
    
    # Testing
    # load best chkpt
    model = CXP_Model()
    model = model.to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)

    # Load the checkpoint
    checkpoint = torch.load(out_dir / 'cxp_pneu_densenet.chkpt')

    # Load the EMA model state
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

    # Set to eval mode
    ema_model.eval()
    
    # Verify that I am reproducing my earlier val results with this reloaded model
    logging.info(f"Best val AUROC (from training): {best_val_auroc:.4f}")

    # Re-run on val set with loaded EMA model
    ema_model.eval()
    val_auroc_reloaded = BinaryAUROC()
    val_results = []
    with torch.no_grad():
        for inputs, labels, drain in val_loader:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits, projections = ema_model(inputs)
            logits = logits.reshape(-1)
            val_auroc_reloaded.update(logits, labels)
            val_results.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(logits.cpu()), 'drain': drain}))
    logging.info(f"Val AUROC after reloading: {val_auroc_reloaded.compute():.4f}")     
    val_results_df = pd.concat(val_results, ignore_index=True)
    
    test_loss_aligned = 0.0
    test_bce_aligned = 0.0
    test_supcon_aligned = 0.0
    test_auroc_aligned.reset()
    test_results_aligned = []
    test_brier_sum = 0.0
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_aligned):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits, projections = ema_model(inputs)
            logits = logits.reshape(-1)
            bce_loss = BCE(logits, labels.to(torch.float32))
            supcon_loss = supconloss(projections, labels)

            if supcon:
                loss = bce_loss + LAMBDA_SUPCON * supcon_loss
            else:
                loss = bce_loss

            test_loss_aligned += loss.item() * inputs.size(0)
            test_bce_aligned += bce_loss.item() * inputs.size(0)
            test_supcon_aligned += supcon_loss.item() * inputs.size(0)
            test_auroc_aligned.update(logits, labels)
            test_results_aligned.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(logits.cpu()), 'drain': drain}))
            # Compute Brier Score: mean((probs - labels)^2)
            probs = torch.sigmoid(logits)
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_sum += brier                 
            
    test_loss_aligned /= len(test_data_aligned)  
    test_bce_aligned /= len(test_data_aligned)
    test_supcon_aligned /= len(test_data_aligned)
    test_brier_aligned = test_brier_sum / len(test_data_aligned)
    test_results_aligned_df = pd.concat(test_results_aligned, ignore_index=True)
    test_results_aligned_df.label = test_results_aligned_df.label.astype(bool)
    test_results_aligned_df.y_prob = test_results_aligned_df.y_prob.astype(np.float64)
    test_results_aligned_df.to_csv(out_dir / 'cxp_pneu_densenet_test_results_aligned.csv')
    logging.info(f"Test Loss ALIGNED: {test_loss_aligned:.4f} AUROC: {test_auroc_aligned.compute():.4f} Brier: {test_brier_aligned:.4f}\n")
    
    test_loss_misaligned = 0.0
    test_bce_misaligned = 0.0
    test_supcon_misaligned = 0.0
    test_auroc_misaligned.reset()
    test_results_misaligned = []
    test_brier_sum = 0.0
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_misaligned):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            logits, projections = ema_model(inputs)
            logits = logits.reshape(-1)
            bce_loss = BCE(logits, labels.to(torch.float32))
            supcon_loss = supconloss(projections, labels)

            if supcon:
                loss = bce_loss + LAMBDA_SUPCON * supcon_loss
            else:
                loss = bce_loss
            
            test_loss_misaligned += loss.item() * inputs.size(0)
            test_bce_misaligned += loss.item() * inputs.size(0)
            test_supcon_misaligned += loss.item() * inputs.size(0)
            test_auroc_misaligned.update(logits, labels)
            test_results_misaligned.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(logits.cpu()), 'drain': drain}))
            # Compute Brier Score: mean((probs - labels)^2)
            probs = torch.sigmoid(logits)
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_sum += brier                     
            
    test_loss_misaligned /= len(test_data_misaligned)  
    test_bce_misaligned /= len(test_data_misaligned)
    test_supcon_misaligned /= len(test_data_misaligned)
    test_brier_misaligned = test_brier_sum / len(test_data_misaligned)
    test_results_misaligned_df = pd.concat(test_results_misaligned, ignore_index=True)
    test_results_misaligned_df.label = test_results_misaligned_df.label.astype(bool)
    test_results_misaligned_df.y_prob = test_results_misaligned_df.y_prob.astype(np.float64)
    test_results_misaligned_df.to_csv(out_dir / 'cxp_pneu_densenet_test_results_misaligned.csv')
    logging.info(f"Test Loss MISALIGNED: {test_loss_misaligned:.4f} AUROC: {test_auroc_misaligned.compute():.4f} Brier: {test_brier_misaligned:.4f}\n")

    return {'test_auroc_aligned': test_auroc_aligned.compute(),
            'test_auroc_misaligned': test_auroc_misaligned.compute(),
            'test_brier_aligned': test_brier_aligned,
            'test_brier_misaligned': test_brier_misaligned,
            'test_loss_aligned': test_loss_aligned,
            'test_loss_misaligned': test_loss_misaligned}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False,
                       help='Directory above /CheXpert-v1.0-small',
                       default='/data')
    parser.add_argument('--csv_dir', type=str, required=False,
                       help='Directory above that contains train_drain_shortcut.csv, etc.',
                       default='.')                 
    parser.add_argument('--out_dir', type=str, required=False,
                       help='Directory where outputs (logs, checkpoints, plots, etc.) will be placed',
                       default='~/cxp_shortcut_out')     
    parser.add_argument('--balance_train', type=lambda x: x.lower() == 'true', required=False,
                       help='Whether to balance the train set by drain presence for both classes',
                       default=False)             
    parser.add_argument('--balance_val', type=lambda x: x.lower() == 'true', required=False,
                       help='Whether to balance the val set by drain presence for both classes',
                       default=False)                 
    parser.add_argument('--select_chkpt_on', type=str, required=False,
                       help='Val metric to select final chkpt on. Possible values right now: AUROC, Loss.',
                       default="loss")            
    parser.add_argument('--supcon', type=lambda x: x.lower() == 'true', required=False,
                       help='Use supervised contrastive learning?',
                       default=False)                           
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    
    setup_logging(out_dir)

    if torch.cuda.is_available():
        logging.info("Using GPU")
    else:
        logging.info("Using CPU")

    if args.balance_train:
        logging.info("Using train set balancing")
    else:
        logging.info("Using native train distribution = no balancing")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    results = []
    for _ in range(NUM_RUNS):
        setup_wandb(out_dir, args.balance_train, args.balance_val, args.select_chkpt_on, args.supcon) 
        results.append(train_and_eval(data_dir, csv_dir, out_dir, 
                                      balance_train_set=args.balance_train,
                                      balance_val_set=args.balance_val,
                                      select_chkpt_on=args.select_chkpt_on,
                                      supcon=args.supcon))
        wandb.finish()

    logging.info("===== FINAL RESULTS ACROSS RUNS =====")
    for metric in results[0].keys():
        vals = [result[metric] for result in results]
        logging.info(f'{metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')