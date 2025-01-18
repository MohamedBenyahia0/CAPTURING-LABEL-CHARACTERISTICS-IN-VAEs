""" 
Mohamed wrote the background, the results and discussion parts, 
implemented and tested the first version of model with the non-split latent space.
Tancrede made the division of the latent space, tested the adapted model, 
compared the beta influence with the results of a non-separated latent, 
and wrote the model part in the report. 
Hadrien wrote the experiments and implementation part and worked on the graph 
in the poster, optimized and improved the model after running a lot of tests 
with different  values for the hyperparameters.
"""
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data
from losses import compute_ccvae_loss
from models import CCVAE
from train import train_ccvae
from evaluate import evaluate_classification,generate_cond_samples,test_reconstruction,perform_intervention
from utils import plot_train_valid_losses,plot_valid_accuracy,load_model,plot_generated_images,plot_intervened_images
loaders=load_and_preprocess_data(batch_size=64)
model=CCVAE(latent_dim=28,hidden_dim=64,output_dim=1,num_labels=10)
beta= 0.4
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
history=train_ccvae(model,loaders,optimizer,compute_ccvae_loss,epochs=60,device=device,beta= beta)
plot_train_valid_losses(history['train_loss'],history['val_loss'],save_path="train_val_loss.png")
plot_valid_accuracy(history['val_acc'],save_path='valid_accuracy_png')
model_path='./saved_models/ccvae_weights.pth'
model = load_model(model, model_path, device)
test_loss, test_accuracy = evaluate_classification(
    model.to(device), loaders['test'], compute_ccvae_loss, beta=beta,device=device)

# Report results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")


model.eval()
#Test Reconstruction of images
test_reconstruction(model, loaders['test'], device='cuda', n_images=10)

# Generate conditional samples
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(device)
generated_images = generate_cond_samples(model, labels, device)
plot_generated_images(generated_images, num_images=10,img_size=(28, 28),save_path='condition_generated_images.png')
# Generate latent intervention 
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
intervened_images=perform_intervention(model,alphas, label_1=0, label_2=1, device=device)
plot_intervened_images(intervened_images, alphas=alphas,num_images=10,img_size=(28, 28),save_path='intervened_images.png')