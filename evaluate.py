import torch
import torch.nn as nn
import numpy as np
import torch.distributions as td
from utils import plot_reconstructions
def evaluate_classification(model, dataloader, loss_function,beta,device) :
    """
    Evaluate classification loss and accuracy on the test set.
    """
    test_loss = 0
    correct = 0
    total = 0
    d=model.encoder.latent_dim
    num_classes=10

    for inputs, targets in dataloader:
        inputs = inputs.to(device)

        
        targets = targets.to(device)
        encoded_batch = model.encoder(inputs)
        mu_q = encoded_batch[:, :d]
        sigma_q = nn.Softplus()(encoded_batch[:, d:])
        q_zgivenx = td.Independent(td.Normal(loc=mu_q, scale=sigma_q), reinterpreted_batch_ndims=1)
        z = q_zgivenx.rsample()
        z_c = z[: ,:num_classes]
        

        

        # Supervised case: use the predicted labels for reconstruction loss
        y_pred = model.classifier(z_c)
        p_xgivenz = td.Independent(td.Bernoulli(logits = model.decoder(z)), reinterpreted_batch_ndims = 1)
        locs_p_zc, scales_p_zc = model.cond_prior(targets)  # Conditional prior given labels
        

        loss = loss_function(True, inputs,targets,mu_q, sigma_q,locs_p_zc, scales_p_zc,y_pred,model, num_classes, device,beta)
        test_loss += loss.item()

        # Calculate accuracy for supervised validation
        _, predicted = torch.max(y_pred, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return test_loss/len(dataloader),100 * correct / total



def reconstruct_image(model, x, device):
    """
    Reconstruct an image from its latent space representation.
    
    Args:
        model: The trained model with encoder and decoder.
        x: The input image to reconstruct.
        device: The device to run the model on (cpu or cuda).
    
    Returns:
        reconstructed_image: The reconstructed image.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Move the input image to the device
    x = x.to(device)
    d=model.encoder.latent_dim

    with torch.no_grad():  # Disable gradient computation for inference
        # Pass the image through the encoder to get the latent variables (mean and std)
        encoded_batch = model.encoder(x)
        mu_q = encoded_batch[:, :d]
        sigma_q = nn.Softplus()(encoded_batch[:, d:])
        
        # Sample from the posterior distribution q(z|x) (latent representation)
        p_z = torch.distributions.Normal(mu_q, sigma_q)
        z_samples = p_z.rsample()  # Sample from the distribution
        
        # Reconstruct the image using the decoder
        p_xgivenz = td.Independent(td.Bernoulli(logits = model.decoder(z_samples)), reinterpreted_batch_ndims = 1)
        reconstructed_image = p_xgivenz.sample()
        
        
        
    return reconstructed_image.cpu().numpy()  # Convert to numpy for visualization

def test_reconstruction(model, test_loader, device='cuda', n_images=10):
    """
    Test the reconstruction function with images from the test_loader.
    
    Args:
        model: The trained model with encoder and decoder.
        test_loader: The data loader for the test dataset.
        device: The device to run the model on (cpu or cuda).
        n_images: Number of images to reconstruct and display.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Get a batch of images from the test_loader
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i == 0:  # Just use the first batch
                # Select the first n_images from the batch
                x = x[:n_images].to(device)
                
                # Reconstruct the images
                reconstructed_images = reconstruct_image(model, x, device)
                
                # Plot the original and reconstructed images
                plot_reconstructions(x, reconstructed_images, n_images,save_path='reconstructed_images.png')
                break  # Only process the first batch            



def generate_cond_samples(model, labels: torch.Tensor, device: str) -> torch.Tensor:
    """
    Perform conditional generation given a set of labels.
    """
    model.eval()
    labels.to(device)
    
    
    with torch.no_grad():
        # Get conditional prior parameters for z_c
        locs_p_zc, scales_p_zc = model.cond_prior(labels)  # Conditional prior given labels
        #Sample
        p_zc = td.Independent(td.Normal(loc=locs_p_zc, scale=scales_p_zc), reinterpreted_batch_ndims=1)
        z_c = p_zc.sample().to(device)
        batch = z_c.size(0)
        p_zs = td.Normal(*model._z_prior_params((batch,model.style_dim)))
        z_s = p_zs.sample()
        z_s = z_s.repeat((z_c.size(0),1)).to(device)

        z = torch.stack([torch.cat((u_i, v_i)) for u_i, v_i in zip(z_c, z_s)])
        # Decode z to generate images
        p_xgivenz = td.Independent(td.Bernoulli(logits = model.decoder(z)), reinterpreted_batch_ndims = 1)
        generated_images=p_xgivenz.sample()
        
    return generated_images

def generate_latent_intervention(model, label_1, label_2, alpha, device) :
    """
    Perform an intervention by interpolating between the latent vectors of two labels.
    """
    model.eval()
    
    # Convert labels to tensor and move to device
    label_tensor_1 = torch.tensor([label_1], device=device)
    label_tensor_2 = torch.tensor([label_2], device=device)
    
    with torch.no_grad():
        # Get conditional prior parameters for z_c for both labels
        locs_p_zc_1, scales_p_zc_1 = model.cond_prior(label_tensor_1)  # Conditional prior for label_1
        locs_p_zc_2, scales_p_zc_2 = model.cond_prior(label_tensor_2)  # Conditional prior for label_2
        
        # Sample from both conditional priors
        p_zc_1 = td.Independent(td.Normal(loc=locs_p_zc_1, scale=scales_p_zc_1), reinterpreted_batch_ndims=1)
        p_zc_2 = td.Independent(td.Normal(loc=locs_p_zc_2, scale=scales_p_zc_2), reinterpreted_batch_ndims=1)

        z_1 = p_zc_1.sample()  # Latent vector for label_1
        z_2 = p_zc_2.sample()  # Latent vector for label_2

        #compute z_s
        batch = z_1.size(0)
        p_zs = td.Normal(*model._z_prior_params((batch,model.style_dim)))
        z_s = p_zs.sample().to(device)
        
        #Align z_c and z_s
        z_1 = torch.cat((z_1,z_s),dim=1)
        z_2 = torch.cat((z_2,z_s),dim=1)
        # Interpolate between the two latent vectors
        z_interpolated = (1 - alpha) * z_1 + alpha * z_2  # Interpolation

        # Decode the interpolated latent vector into image logits
        p_xgivenz = td.Independent(td.Bernoulli(logits=model.decoder(z_interpolated)), reinterpreted_batch_ndims=1)
        generated_image = p_xgivenz.sample()
        
    return generated_image

def perform_intervention(model,alphas, label_1, label_2, device):
    
    intervened_images=[]
    for i, alpha in enumerate(alphas):
        # Generate the intervened image for the current alpha
        intervened_image = generate_latent_intervention(model, label_1, label_2, alpha, device)
        intervened_images.append(intervened_image)
    intervened_images = torch.stack(intervened_images) if len(intervened_images) > 1 else intervened_images[0]
    return intervened_images
