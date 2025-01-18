import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
from tqdm import tqdm
from utils import save_model
def train_ccvae(model, loaders, optimizer, loss_function, epochs, device, beta=0.2):
    """
    Train the CCVAE model in a semi-supervised setting.
    Args:
        model: The CCVAE model combining Encoder, Decoder, and Classifier.
        loaders : Dict of 'sup' and 'unsup' train data loaders and valid and test data loader
        optimizer: Optimizer for updating model weights.
        loss_function: CCVAE loss function.
        epochs: Number of training epochs.
        device: Device to run the model on ('cpu' or 'cuda').
        beta: Weighting term for KL divergence.
    Returns:
        history: Dictionary containing training and validation loss per epoch.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}  # Track losses and accuracy
    model.to(device)
    d = model.encoder.latent_dim
    num_classes=10
    train_loader_sup = loaders['sup']
    train_loader_unsup = loaders['unsup']
    val_loader = loaders['valid']
    
    for epoch in tqdm(range(epochs)):
        model.train()  # Set model to training mode
        cumulative_loss = 0
        total_samples=0
        

        # Initialize data iterators
        sup_iter = iter(train_loader_sup)
        unsup_iter = iter(train_loader_unsup)
        batches_per_epoch = max(len(train_loader_sup), len(train_loader_unsup))

        for batch_idx in range(batches_per_epoch):
            if batch_idx % 2 == 0 and len(train_loader_sup) > 0:  # Use a supervised batch
                sup = True
                try:
                    inputs, targets = next(sup_iter)
                except StopIteration:
                    sup_iter = iter(train_loader_sup)
                    inputs, targets = next(sup_iter)

                inputs, targets = inputs.to(device), targets.to(device)
                
            else:  # Use an unsupervised batch
                sup = False
                try:
                    inputs, _ = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(train_loader_unsup)
                    inputs, _ = next(unsup_iter)

                inputs = inputs.to(device)

            # Forward pass and compute loss
            encoded_batch = model.encoder(inputs)
            mu_q = encoded_batch[:, :d]
            sigma_q = nn.Softplus()(encoded_batch[:, d:])
            
            
            
            
            if sup:
                # Supervised case: use true labels
                q_zgivenx = td.Independent(td.Normal(loc=mu_q, scale=sigma_q), reinterpreted_batch_ndims=1)
                z = q_zgivenx.rsample()

                y_pred = model.classifier(z)

                locs_p_zc, scales_p_zc = model.cond_prior(targets)  # Conditional prior given labels
                

                loss = loss_function(True, inputs,targets,mu_q, sigma_q,locs_p_zc, scales_p_zc,y_pred,model, num_classes, device,beta)
            else:
                # Unsupervised case: no labels, just reconstruct images
                loss = loss_function(sup=False, x=inputs, y=None, mu_q=mu_q, sigma_q=sigma_q, 
                mu_p=None, sigma_p=None, y_pred=None, model=model, num_classes=num_classes, device=device,beta=beta)
            

            cumulative_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = cumulative_loss / total_samples
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                encoded_batch = model.encoder(inputs)
                mu_q = encoded_batch[:, :d]
                sigma_q = nn.Softplus()(encoded_batch[:, d:])
                q_zgivenx = td.Independent(td.Normal(loc=mu_q, scale=sigma_q), reinterpreted_batch_ndims=1)
                z = q_zgivenx.rsample()

                
                

                # Supervised case: use the predicted labels for reconstruction loss
                y_pred = model.classifier(z)
                p_xgivenz = td.Independent(td.Bernoulli(logits = model.decoder(z)), reinterpreted_batch_ndims = 1)
                locs_p_zc, scales_p_zc = model.cond_prior(targets)  # Conditional prior given labels
                

                loss = loss_function(True, inputs,targets,mu_q, sigma_q,locs_p_zc, scales_p_zc,y_pred,model, num_classes, device,beta)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(y_pred, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)

        
            

        # Print epoch results
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}")
    save_model(model, path=f"saved_models/ccvae_weights.pth", optimizer=optimizer)

    return history
