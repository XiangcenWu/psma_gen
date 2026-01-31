import torch


def sample_labels_to_binary(mask):
    
    binary_mask = (mask != 0).to(mask.dtype)

    return binary_mask



if __name__ == '__main__':

    mask = torch.randint(0, 128, (128, 128 , 128))
    binary_mask, chosen = sample_labels_to_binary(mask, num_samples=5)

    print("Chosen labels:", chosen)
    print("Binary mask unique values:", torch.unique(binary_mask))
