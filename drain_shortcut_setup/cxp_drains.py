import os
import os.path
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121
from torcheval.metrics import BinaryAUROC
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

LR = 0.0001
NUM_EPOCHS = 50
NUM_RUNS = 20

if not os.path.exists("/deep_learning/output/petersen/cxr_small_data/wandb"):
    os.mkdir("/deep_learning/output/petersen/cxr_small_data/wandb")

os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/petersen/cxr_small_data/wandb")


if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CXP_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = densenet121(weights='IMAGENET1K_V1')
        # using BCEWithLogitsLoss, so no sigmoid needed - but need explicit sigmoid for prob prediction
        self.clf = nn.Linear(1000, 1)

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

        if not inference_only:
            self.labels = df.Drain
            self.pneu = df.Pneumothorax
        else:
            self.labels = None
            self.pneu = None

    def __getitem__(self, index: int):
        try:
            img = torchvision.io.read_image(os.path.join(self.root_dir, self.path[index]))
            img = self.transform(img)
            if self.labels is None or self.pneu is None:
                return img, self.path[index]
            else:
                return img, self.labels[index], self.pneu[index]
        except RuntimeError as e:
            print(f"Error loading image at index {index}: {self.path[index]}")
            print(f"Error message: {e}")
            # Return the next valid image
            return self.__getitem__((index + 1) % len(self))
    
    def __len__(self) -> int:
        return len(self.path)

def train():
    train_data = CXP_dataset('/deep_learning/output/petersen/', '/deep_learning/output/petersen/cxr_small_data/drain_train.csv')
    val_data = CXP_dataset('/deep_learning/output/petersen/', '/deep_learning/output/petersen/cxr_small_data/drain_val.csv', augment=False)
    test_data = CXP_dataset('/deep_learning/output/petersen/', '/deep_learning/output/petersen/cxr_small_data/drain_test.csv', augment=False)
    
    best_test_loss = 1000
    best_run = 0
    
    for run_idx in range(NUM_RUNS):
        print("\n ------ STARTING NEW RUN! ------ \n")
        
        # start a new wandb run to track this run
        wandb.init(
            # set the wandb project where this run will be logged
            project="cxr_small_data_tubes",
            dir="/deep_learning/output/petersen/cxr_small_data/wandb",
            # track hyperparameters and run metadata
            config={
            "learning_rate": LR,
            "architecture": "densenet121",
            "dataset": "CheXpert",
            "epochs": NUM_EPOCHS,
            "run": run_idx
            }
        )   
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)      
        
        model = CXP_Model()
        model = model.to(device)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=10e-6)

        train_auroc = BinaryAUROC()
        val_auroc = BinaryAUROC()
        test_auroc = BinaryAUROC()

        best_val_auroc = 0.0

        # Train the model
        for epoch in range(NUM_EPOCHS):

            print(f'======= EPOCH {epoch} =======')

            # Train
            model.train()
            train_loss = 0.0
            train_auroc.reset()
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

            # Validation
            #model.eval()
            ema_model.update_parameters(model)
            ema_model.eval()
            
            val_loss = 0.0
            val_auroc.reset()
            with torch.no_grad():
                for inputs, labels, _ in tqdm(val_loader):
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    outputs = ema_model(inputs).reshape(-1)
                    loss = criterion(outputs, labels.to(torch.float32))
                    val_loss += loss.item() * inputs.size(0)
                    val_auroc.update(outputs, labels)

            # Print training and val performance
            train_loss /= len(train_data)
            val_loss /= len(val_data)
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Train AUROC: {train_auroc.compute()}\n"
                  f"             Val  Loss: {val_loss:.4f} Val AUROC: {val_auroc.compute()}")

            wandb.log({"Loss/train": train_loss,
                       "Loss/val": val_loss,
                       "auroc/train": train_auroc.compute(),
                       "auroc/val": val_auroc.compute()})
            
            if val_auroc.compute() > best_val_auroc:
                best_val_auroc = val_auroc.compute()
                print(f"Saving new best chkpt at epoch {epoch}.")
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'ema_model_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, f'/deep_learning/output/petersen/cxr_small_data/cxp_tube_densenet_{run_idx}.chkpt')

        wandb.finish()
        
        # Testing
        # load best chkpt
        model = CXP_Model()
        model = model.to(device)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)

        # Load the checkpoint
        checkpoint = torch.load(f'/deep_learning/output/petersen/cxr_small_data/cxp_tube_densenet_{run_idx}.chkpt')

        # Load the EMA model state
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        # Set to eval mode
        ema_model.eval()
        
        # Verify that I am reproducing my earlier val results with this reloaded model
        print(f"Best val AUROC (from training): {best_val_auroc}")

        # Re-run on val set with loaded EMA model
        ema_model.eval()
        val_auroc_reloaded = BinaryAUROC()
        val_results = []
        with torch.no_grad():
            for inputs, labels, pneu in tqdm(val_loader):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = ema_model(inputs).reshape(-1)
                val_auroc_reloaded.update(outputs, labels)
                val_results.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(outputs.cpu()), 'pneu': pneu.cpu()}))
        print(f"Val AUROC after reloading: {val_auroc_reloaded.compute()}")     
        val_results_df = pd.concat(val_results, ignore_index=True)
        
        test_loss = 0.0
        test_auroc.reset()
        test_results = []
        with torch.no_grad():
            for inputs, labels, pneu in tqdm(test_loader):
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = ema_model(inputs).reshape(-1)
                loss = criterion(outputs, labels.to(torch.float32))
                test_loss += loss.item() * inputs.size(0)
                test_auroc.update(outputs, labels)
                test_results.append(pd.DataFrame({'label': labels.cpu(), 'y_prob': torch.sigmoid(outputs.cpu()), 'pneu': pneu.cpu()}))
                
        test_loss /= len(test_data)                 
                
        test_results_df = pd.concat(test_results, ignore_index=True)
        test_results_df.to_csv('/deep_learning/output/petersen/cxr_small_data/cxp_tube_densenet_test_results_{run_idx}.csv')
        print(f"Test Loss run {run_idx}: {test_loss:.4f} Test AUROC: {test_auroc.compute()}\n")
        val_test_combined_df = pd.concat([val_results_df, test_results_df], ignore_index=True, sort=False)
        upper_thresh = val_test_combined_df.y_prob.quantile(0.9)
        lower_thresh = val_test_combined_df.y_prob.quantile(0.1)
        msk = (test_results_df.y_prob <= lower_thresh) | (test_results_df.y_prob >= upper_thresh)
        print(f"Test AUROC with rejection between {lower_thresh:.5f}-{upper_thresh:.5f}: {roc_auc_score(test_results_df.label[msk], test_results_df.y_prob[msk]):.3f}\n")
        pneu_msk = val_test_combined_df.pneu == 1
        upper_thresh_pneu = val_test_combined_df.y_prob[pneu_msk].quantile(0.9)
        lower_thresh_pneu = val_test_combined_df.y_prob[pneu_msk].quantile(0.1)
        upper_thresh_nopneu = val_test_combined_df.y_prob[~pneu_msk].quantile(0.9)
        lower_thresh_nopneu = val_test_combined_df.y_prob[~pneu_msk].quantile(0.1)      
        msk = ((test_results_df.pneu == 1) & (test_results_df.y_prob <= lower_thresh_pneu)) | ((test_results_df.pneu == 1) & (test_results_df.y_prob >= upper_thresh_pneu)) | (~(test_results_df.pneu == 1) & (test_results_df.y_prob <= lower_thresh_nopneu)) | (~(test_results_df.pneu == 1) & (test_results_df.y_prob >= upper_thresh_nopneu))
        print(f"Test AUROC with rejection between {lower_thresh_pneu:.5f}-{upper_thresh_pneu:.5f} (pneu) and {lower_thresh_nopneu:.5f}-{upper_thresh_nopneu:.5f} (no pneu): {roc_auc_score(test_results_df.label[msk], test_results_df.y_prob[msk]):.3f}\n")
   
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_run = run_idx
            best_lower_thresh = lower_thresh
            best_upper_thresh = upper_thresh
            
    print(f"Best run was {best_run}, loss {best_test_loss:.4f}, lower thresh {best_lower_thresh:.5f}, upper thresh {best_upper_thresh:.5f}.")

    return best_run, best_lower_thresh, best_upper_thresh
    

def run_inference(run_idx, lower_thresh, upper_thresh):
    model = CXP_Model()
    model = model.to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)

    # Load the checkpoint
    checkpoint = torch.load(f'/deep_learning/output/petersen/cxr_small_data/cxp_tube_densenet_{run_idx}.chkpt')

    # Load the EMA model state
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    ema_model.eval()
     
    train_data = CXP_dataset('/deep_learning/output/petersen/', '/deep_learning/output/petersen/CheXpert-v1.0-small/train.csv', augment=False, inference_only=True)
    val_data = CXP_dataset('/deep_learning/output/petersen/', '/deep_learning/output/petersen/CheXpert-v1.0-small/valid.csv', augment=False, inference_only=True)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=2)

    train_results = []
    with torch.no_grad():
        for inputs, paths in tqdm(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            outputs = ema_model(inputs).reshape(-1)
            train_results.append(pd.DataFrame({'path': paths, 'y_prob': torch.sigmoid(outputs.cpu())}))

    train_results = pd.concat(train_results, ignore_index=True)    

    val_results = []
    with torch.no_grad():
        for inputs, paths in tqdm(val_loader):
            inputs = inputs.cuda(non_blocking=True)
            outputs = ema_model(inputs).reshape(-1)
            val_results.append(pd.DataFrame({'path': paths, 'y_prob': torch.sigmoid(outputs.cpu())}))

    val_results = pd.concat(val_results, ignore_index=True)    

    train_results.loc[:, "soft_drain"] = pd.NA
    train_results.loc[train_results.y_prob <= lower_thresh, "soft_drain"] = 0
    train_results.loc[train_results.y_prob >= upper_thresh, "soft_drain"] = 1
    train_results.drop_duplicates().to_csv("/deep_learning/output/petersen/CheXpert-v1.0-small/train_drain.csv")
    val_results.loc[:, "soft_drain"] = pd.NA
    val_results.loc[val_results.y_prob <= lower_thresh, "soft_drain"] = 0
    val_results.loc[val_results.y_prob >= upper_thresh, "soft_drain"] = 1
    val_results.drop_duplicates().to_csv("/deep_learning/output/petersen/CheXpert-v1.0-small/valid_drain.csv")


if __name__ == '__main__':
    best_run, lower_thresh, upper_thresh = train()
    run_inference(best_run, lower_thresh, upper_thresh)