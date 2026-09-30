import torch
import torch.nn as nn
import torch.nn.functional as F

# Single-GPU version adapted by Claude from https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py
# (or at least I told it to)
# https://claude.ai/share/ebb8e3c5-ae51-447d-b1e9-034e80dabdea

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for binary/multi-class classification.
    Adapted from StableRep's MultiPosConLoss:
    https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py
    
    This is a single-GPU version (no distributed training).
    """
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, feats, labels):
        """
        Args:
            feats: Feature embeddings, shape [B, D] (should be from a projection head)
            labels: Class labels, shape [B] (0 or 1 for binary classification)
        
        Returns:
            loss: Scalar supervised contrastive loss
        """
        device = feats.device
        batch_size = feats.size(0)
        
        # L2 normalize features
        feats = F.normalize(feats, dim=-1, p=2)
        
        # Compute similarity matrix: [B, B]
        similarity_matrix = torch.matmul(feats, feats.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        # mask[i, j] = 1 if labels[i] == labels[j], else 0
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrasts (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have at least one positive pair
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss