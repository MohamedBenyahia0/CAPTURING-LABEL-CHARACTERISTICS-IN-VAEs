import torch
import torch.nn as nn  
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
class Encoder(nn.Module):
    def __init__(self, latent_dim: int, num_classes : int):
        """
        Define the encoder for CCVAE.
        Args:
            
            latent_dim: Size of latent space is latent_dim*2
            because we output both the mean of q and the diagonal of the covariance
        """
        super(Encoder, self).__init__()
        self.hidden_dim = 64
        self.latent_dim=latent_dim
        self.num_classes= num_classes
        '''
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),  # Input: (1, 28, 28), Output: (32, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),  # Output: (64, 7, 7)
            nn.ReLU(True),
            nn.Conv2d(64, self.latent_dim*2, 7, 1),  # Output: (latent_dim*2, 1, 1)
            nn.ReLU(True)
        )

        '''
        self.encoder= nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(True),
            View((-1,32*2*2)),
            nn.Linear(32*2*2, 64),
            nn.ReLU(True),
            nn.Linear(64, 2*self.latent_dim),
            nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.
        """
        z=self.encoder(x)
        z=z.view(-1,self.latent_dim*2)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int ,output_dim: int):
        """
        Define the decoder for CCVAE.
        Args:
            latent_dim: Size of latent space is latent_dim*2
            because we output both the mean of q and the diagonal of the covariance
            hidden_dim : hidden dimension
            output_dim: Dimensionality of reconstructed images.
        """
        super(Decoder, self).__init__()
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        '''
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim * 7 * 7),  # Project latent vector to 7x7xhidden_dim
            View((-1, hidden_dim, 7, 7)),  # Reshape to image dimensions
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.output_dim, 3, stride=1, padding=1),  # No upsampling, just channel adjustment
        
        )
        '''
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),  # Project latent vector to 7x7xhidden_dim
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim * 7 * 7),
            View((-1, hidden_dim, 7, 7)),  # Reshape to image dimensions
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.output_dim, 3, stride=1, padding=1),  # No upsampling, just channel adjustment
        
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.
        """
        out=self.decoder(z)
        return out
        

class Classifier(nn.Module):
    def __init__(self, latent_dim: int, num_labels: int):
        """
        Define the classifier for predicting labels.
        Args:
            latent_dim: Size of latent space used for labels.
            num_labels: Number of label classes.
        """
        super(Classifier, self).__init__()
        self.latent_dim=latent_dim
        self.num_labels=num_labels
        
        self.output_layer = nn.Linear(self.num_labels, self.num_labels)
      

        

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.
        """
        
        # Map to class logits
        out=self.output_layer(z[:,:self.num_labels])
        
        
        return out
    
class CondPrior(nn.Module):
    def __init__(self, dim, num_classes):
        super(CondPrior, self).__init__()
        self.dim = dim
        self.num_classes = num_classes

        # Define a simple neural network to predict the prior distribution parameters based on class labels
        self.fc = nn.Sequential(
            nn.Embedding(num_classes, 32),        # Embedding layer for the class labels
            nn.ReLU(),
            nn.Linear(32, 2 * dim)                # Output both mean and scale parameters
        )

    def forward(self, labels):
        """
        Given labels, produce conditional prior (mean, scale).
        """
        # Get prior parameters from the embedding network
        prior_params = self.fc(labels)
        
        # Split the output into mean and scale
        locs_p_zc, scales_p_zc = prior_params.chunk(2, dim=-1)

        # Apply a softplus to scale to ensure it is positive
        scales_p_zc = F.softplus(scales_p_zc)
        
        return locs_p_zc, scales_p_zc

class CCVAE(nn.Module):
    def __init__(self,latent_dim:int,hidden_dim:int,output_dim:int,num_labels:int):
        super(CCVAE, self).__init__()
        
        self.latent_dim=latent_dim
        self.style_dim = latent_dim-num_labels
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_labels=num_labels
        self.cond_prior=CondPrior(self.num_labels,self.num_labels)
        self.encoder=Encoder(self.latent_dim, self.num_labels)
        self.decoder=Decoder(self.latent_dim,self.hidden_dim,self.output_dim)
        self.classifier=Classifier(self.num_labels,self.num_labels)
        
    def _z_prior_params(self, shape):
        ones = torch.ones(shape)
        zeros = torch.zeros(shape)
        return zeros, ones

        
