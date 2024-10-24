import torch.nn.functional as F

def batch_all_triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Compute triplet loss using the batch all strategy.

    This strategy involves computing the triplet loss for all possible combinations 
    of anchor, positive, and negative samples in a batch.
    """


    distance_matrix = compute_distance_matrix(anchor, positive, negative)
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin)
    loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 2] + margin)
    return torch.mean(loss)