import torch

def batch_hard_triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Compute triplet loss using the batch hard strategy.

    This strategy involves computing the triplet loss only for the 
    hardest negative sample for each anchor-positive pair in a batch.
    """

    
    distance_matrix = compute_distance_matrix(anchor, positive, negative)
    hard_negative = torch.argmax(distance_matrix[:, 2])
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin)
    loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0][hard_negative] - distance_matrix[:, 2] + margin)
    return torch.mean(loss)