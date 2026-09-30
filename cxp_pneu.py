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
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.models import densenet121
from torcheval.metrics import BinaryAUROC

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

LR = 0.0001
NUM_EPOCHS = 30
NUM_RUNS = 10

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
        "epochs": NUM_EPOCHS
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
            nn.Linear(num_features, 512),  # 1024 -> 512
            nn.ReLU(),
            nn.Linear(512, 1)  # 512 -> 1
        )

    def forward(self, x):
        z = self.encode(x)
        return self.clf(z)
    
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

def train_and_eval(data_dir, csv_dir, out_dir):
    train_data = CXP_dataset(data_dir, csv_dir / 'train_drain_shortcut.csv')
    val_data = CXP_dataset(data_dir, csv_dir / 'val_drain_shortcut.csv', augment=False)
    test_data_aligned = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_aligned.csv', augment=False)
    test_data_misaligned = CXP_dataset(data_dir, csv_dir / 'test_drain_shortcut_misaligned.csv', augment=False)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)
    test_loader_aligned = torch.utils.data.DataLoader(test_data_aligned, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)      
    test_loader_misaligned = torch.utils.data.DataLoader(test_data_misaligned, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)      
    
    model = CXP_Model()
    model = model.to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)

    train_auroc = BinaryAUROC()
    val_auroc = BinaryAUROC()
    test_auroc_aligned = BinaryAUROC()
    test_auroc_misaligned = BinaryAUROC()

    best_val_loss = 10000.0

    # Train the model
    for epoch in range(NUM_EPOCHS):

        print(f'======= EPOCH {epoch} =======')

        # Train
        model.train()
        train_loss = 0.0
        train_auroc.reset()
        train_brier_sum = 0.0
        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs).reshape(-1)
            loss = criterion(outputs, labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_auroc.update(outputs, labels)
            # Compute Brier Score: mean((probs - labels)^2)
            probs = torch.sigmoid(outputs)
            brier = ((probs - labels.float()) ** 2).sum().item()
            train_brier_sum += brier            

        # Validation
        #model.eval()
        ema_model.update_parameters(model)
        ema_model.eval()
        
        val_loss = 0.0
        val_auroc.reset()
        val_brier_sum = 0.0
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = ema_model(inputs).reshape(-1)
                loss = criterion(outputs, labels.to(torch.float32))
                val_loss += loss.item() * inputs.size(0)
                val_auroc.update(outputs, labels)
                # Compute Brier Score: mean((probs - labels)^2)
                probs = torch.sigmoid(outputs)
                brier = ((probs - labels.float()) ** 2).sum().item()
                val_brier_sum += brier                  

        # Print training and val performance
        train_loss /= len(train_data)
        val_loss /= len(val_data)
        train_brier = train_brier_sum / len(train_data)
        val_brier = val_brier_sum / len(val_data)
        logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} AUROC: {train_auroc.compute():.4f} Brier: {train_brier:.4f}\n"
              f"             Val  Loss: {val_loss:.4f} AUROC: {val_auroc.compute():.4f} Brier: {val_brier:.4f}")

        wandb.log({"Loss/train": train_loss,
                   "Loss/val": val_loss,
                   "auroc/train": train_auroc.compute(),
                   "auroc/val": val_auroc.compute(),
                   "brier/train": train_brier,
                   "brier/val": val_brier})
        
        if val_loss < best_val_loss:  # val_auroc.compute() > best_val_auroc:
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

    wandb.finish()
    
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
            outputs = ema_model(inputs).reshape(-1)
            val_auroc_reloaded.update(outputs, labels)
            val_results.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(outputs.cpu()), 'drain': drain}))
    logging.info(f"Val AUROC after reloading: {val_auroc_reloaded.compute():.4f}")     
    val_results_df = pd.concat(val_results, ignore_index=True)
    
    test_loss_aligned = 0.0
    test_auroc_aligned.reset()
    test_results_aligned = []
    test_brier_sum = 0.0
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_aligned):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = ema_model(inputs).reshape(-1)
            loss = criterion(outputs, labels.to(torch.float32))
            test_loss_aligned += loss.item() * inputs.size(0)
            test_auroc_aligned.update(outputs, labels)
            test_results_aligned.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(outputs.cpu()), 'drain': drain}))
            # Compute Brier Score: mean((probs - labels)^2)
            probs = torch.sigmoid(outputs)
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_sum += brier                 
            
    test_loss_aligned /= len(test_data_aligned)  
    test_brier_aligned = test_brier_sum / len(test_data_aligned)
    test_results_aligned_df = pd.concat(test_results_aligned, ignore_index=True)
    test_results_aligned_df.label = test_results_aligned_df.label.astype(bool)
    test_results_aligned_df.y_prob = test_results_aligned_df.y_prob.astype(np.float64)
    test_results_aligned_df.to_csv(out_dir / 'cxp_pneu_densenet_test_results_aligned.csv')
    logging.info(f"Test Loss ALIGNED: {test_loss_aligned:.4f} AUROC: {test_auroc_aligned.compute():.4f} Brier: {test_brier_aligned:.4f}\n")
    
    test_loss_misaligned = 0.0
    test_auroc_misaligned.reset()
    test_results_misaligned = []
    test_brier_sum = 0.0
    with torch.no_grad():
        for inputs, labels, drain in tqdm(test_loader_misaligned):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = ema_model(inputs).reshape(-1)
            loss = criterion(outputs, labels.to(torch.float32))
            test_loss_misaligned += loss.item() * inputs.size(0)
            test_auroc_misaligned.update(outputs, labels)
            test_results_misaligned.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(outputs.cpu()), 'drain': drain}))
            # Compute Brier Score: mean((probs - labels)^2)
            probs = torch.sigmoid(outputs)
            brier = ((probs - labels.float()) ** 2).sum().item()
            test_brier_sum += brier                     
            
    test_loss_misaligned /= len(test_data_misaligned)  
    test_brier_misaligned = test_brier_sum / len(test_data_misaligned)
    test_results_misaligned_df = pd.concat(test_results_misaligned, ignore_index=True)
    test_results_misaligned_df.label = test_results_misaligned_df.label.astype(bool)
    test_results_misaligned_df.y_prob = test_results_misaligned_df.y_prob.astype(np.float64)
    test_results_misaligned_df.to_csv(out_dir / 'cxp_pneu_densenet_test_results_misaligned.csv')
    logging.info(f"Test Loss MISALIGNED: {test_loss_misaligned:.4f} AUROC: {test_auroc_misaligned.compute():.4f} Brier: {test_brier_misaligned:.4f}\n")
    

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
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    
    setup_logging(out_dir) 
    
    if torch.cuda.is_available():
        logging.info("Using GPU")
    else:
        logging.info("Using CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    train_and_eval(data_dir, csv_dir, out_dir)
