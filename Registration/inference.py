from Registration.mask import sample_labels_to_binary
from Registration.smoothness_losses import l2_gradient



def target_registration_error():
    pass


def inference_batch(
        model, 
        loader,
        optimizer,
        loss_function,
        identity_grid,
        device="cuda:0"
    ):
    
    model.train()
    model.to(device)
    identity_grid.to(device)

    step = 0.
    loss_a = 0.


    for batch in dataloader:


        fdg_pt = batch['fdg_pt'].to(device)
        fdg_mask = batch['fdg_mask'].to(device)

        
        
        psma_pt = batch['psma_pt'].to(device)
        psma_mask = batch['psma_mask'].to(device)


        input = torch.cat([fdg_pt, psma_pt], dim=1)

        # sample mask to be used to train loss
        fdg_mask = sample_labels_to_binary(fdg_mask, mask_per_iteration)
        psma_mask = sample_labels_to_binary(psma_mask, mask_per_iteration)



        ddf = model(input)
        grid = identity_grid + ddf



        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch