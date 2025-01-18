
import torch
import torch.nn as nn
import torch.distributions as td
def compute_ccvae_loss(sup, x, y, mu_q, sigma_q, mu_p, sigma_p, y_pred, 
                       model, num_classes, device, beta=0.5):
    """
    Compute the CCVAE loss for supervised and unsupervised cases.
    
    Args:
        sup: Boolean indicating whether we are in the supervised setting.
        x: Original images (batch).
        y: Ground truth labels (None for unsupervised case).
        mu_q: Mean of the posterior (q(z|x)).
        sigma_q: Standard deviation of the posterior (q(z|x)).
        mu_p: Mean of the prior (p(z|y)).
        sigma_p: Standard deviation of the prior (p(z|y)).
        y_pred: Predicted labels (None for unsupervised case).
        model: The CCVAE model (with encoder, decoder, and classifier).
        num_classes: Total number of possible labels (for marginalization over y).
        device: Device to run the computation (CPU or CUDA).
        beta: Weighting term for KL divergence.

    Returns:
        total_loss: The CCVAE loss (reconstruction + classification + beta * KL).
    """

    # Define distributions for q(z|x)
  
    q_zgivenx = td.Independent(td.Normal(loc=mu_q, scale=sigma_q), reinterpreted_batch_ndims=1)
    

    if sup:
        # Supervised case
        # Reconstruction loss
        
        z = q_zgivenx.rsample()  # Sample latent variable z
        batch = z.size(0)

        #Get prior parameters (zeros and ones) and move them to the correct device
        locs_p_zs, scales_p_zs = model._z_prior_params((batch, model.style_dim))
        locs_p_zs = locs_p_zs.to(device)
        scales_p_zs = scales_p_zs.to(device)

        p_z_s = td.Independent(td.Normal(locs_p_zs, scales_p_zs), reinterpreted_batch_ndims=1)  

        p_xgivenz = td.Independent(td.Bernoulli(logits=model.decoder(z)), reinterpreted_batch_ndims=1)
        loss_recons = -p_xgivenz.log_prob(x).mean()
        
        # Classification loss (cross-entropy)
        loss_class = nn.CrossEntropyLoss()(y_pred, y)

        # Prior p(z_c|y)
        
        p_zgiveny = td.Independent(td.Normal(loc=mu_p, scale=sigma_p), reinterpreted_batch_ndims=1)

        #caracteristic latent 
        q_z_carac_givenx = td.Independent(
        td.Normal(loc=q_zgivenx.mean[:, :num_classes], scale=q_zgivenx.stddev[:, :num_classes]),
        reinterpreted_batch_ndims=1
        )

        #Style latent
        q_z_sty_givenx = td.Independent(
            td.Normal(loc=q_zgivenx.mean[:, num_classes:], scale=q_zgivenx.stddev[:, num_classes:]),
            reinterpreted_batch_ndims=1
            )
        # KL Divergence: q(z_c|x) || p(z_c|y) + q(z\c|x) || q(z_\c)

        loss_KL = td.kl_divergence(q_z_carac_givenx, p_zgiveny).mean()+ td.kl_divergence(q_z_sty_givenx ,p_z_s).mean()
        

    else:
        # Unsupervised case
        # Infer q(y|z_c) using the classifier
        z = q_zgivenx.rsample()  # Sample latent variable z

        batch = z.size(0)
        #Get prior parameters (zeros and ones) and move them to the correct device
        locs_p_z, scales_p_z = model._z_prior_params((batch, model.latent_dim))
        locs_p_z = locs_p_z.to(device)
        scales_p_z = scales_p_z.to(device)

        p_z = td.Independent(td.Normal(locs_p_z, scales_p_z), reinterpreted_batch_ndims=1)  

        y_logits = model.classifier(z)
        q_ygivenzc = td.Categorical(logits=y_logits)

        locs_p_zs, scales_p_zs = model._z_prior_params((batch, model.style_dim))
        locs_p_zs = locs_p_zs.to(device)
        scales_p_zs = scales_p_zs.to(device)

        p_z_s = td.Independent(td.Normal(locs_p_zs, scales_p_zs), reinterpreted_batch_ndims=1) 
        # Loop over all possible labels y
        loss_recons_y = 0
        kl_div_y = 0
        #Style latent
        q_z_sty_givenx = td.Independent(
            td.Normal(loc=q_zgivenx.mean[:, num_classes:], scale=q_zgivenx.stddev[:, num_classes:]),
            reinterpreted_batch_ndims=1
            )
        for label in range(num_classes):
            #Convert to tensor and device
            label_tensor = torch.tensor([label], device=device)

            # Get the conditional prior p(z|y)
            mu_p_label, sigma_p_label = model.cond_prior(label_tensor)
            p_zgiveny = td.Independent(td.Normal(loc=mu_p_label, scale=sigma_p_label), reinterpreted_batch_ndims=1)

            # KL Divergence for this label (per batch element, weighted by q(y|z_c))
            #caracteristic latent 
            q_z_carac_givenx = td.Independent(
            td.Normal(loc=q_zgivenx.mean[:, :num_classes], scale=q_zgivenx.stddev[:, :num_classes]),
            reinterpreted_batch_ndims=1
            )

            locs_p_zs, scales_p_zs = model._z_prior_params((batch, model.style_dim))

            kl_div_label = td.kl_divergence(q_z_carac_givenx, p_zgiveny).mean()  # Shape: [batch_size]
            
        
            kl_div_y += (kl_div_label * q_ygivenzc.probs[:, label]).mean()  # Weighted and averaged

            # Reconstruction loss for this label (per batch element, weighted by q(y|z_c))
            p_xgivenz = td.Independent(td.Bernoulli(logits=model.decoder(z)), reinterpreted_batch_ndims=1)
            recons_loss_label = -p_xgivenz.log_prob(x)  # Shape: [batch_size]
            weight = q_ygivenzc.probs[:, label].unsqueeze(1).unsqueeze(2)
            loss_recons_y += (recons_loss_label * weight).mean()  # Weighted and averaged

        # Average over all possible labels
        loss_recons = loss_recons_y
        loss_KL =td.kl_divergence(q_z_sty_givenx ,p_z_s).mean()

        # Classification loss: Cross-entropy between q(y|z_c) and p(y) (assuming uniform prior)
        log_py = -torch.log(torch.tensor(num_classes, dtype=torch.float32, device=device))
        loss_class = (q_ygivenzc.entropy().mean() - log_py)
        
    # Total loss
    
    total_loss = loss_recons + beta * loss_KL + loss_class

    return total_loss
