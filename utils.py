import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to save a model checkpoint
def save_model(model, path, optimizer=None):
    """
    Saves the model checkpoint to the specified path.
    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path to save the model checkpoint.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save (default: None).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

# Function to load a model checkpoint
def load_model(model, path, device):
    """
    Loads the model checkpoint from the specified path.
    Args:
        model (torch.nn.Module): The model to load.
        path (str): Path to the model checkpoint.
        device (torch.device): The device to load the model onto (e.g., 'cpu' or 'cuda').
    Returns:
        model (torch.nn.Module): The model loaded with the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model


#Function to plot training loss and validation loss
def plot_train_valid_losses(train_loss,val_loss,save_path=None):
    """Plot a training and validation losses over Epochs

    Args:
        train_loss : Training loss values
        val_loss : Validation loss values
        save_path  : Path to save the plot. Defaults to None.
    """
    epochs = list(range(1, len(val_loss) + 1))

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_loss, label='Train loss', marker='o')
    plt.plot(epochs, val_loss, label='Valid loss', marker='s')

    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Show grid and display plot
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
#Function to plot validation accuracy
def plot_valid_accuracy(val_acc,save_path=None):
    """Plot validation accuracy over epochs
    Args:
        
        val_acc : Validation loss values
        save_path  : Path to save the plot. Defaults to None.
    """
    epochs = list(range(1, len(val_acc) + 1))

    # Plotting
    plt.figure(figsize=(10,6))
    
    plt.plot(epochs, val_acc, marker='s')

    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()

    # Show grid and display plot
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
#Function to plot generated_images
def plot_generated_images(images, num_images=10,img_size=(28, 28),save_path=None):
    """
    Plot generated images.
    Args:
        images: Array of generated images.
        img_size: Size of each image (default: 28x28).
    """
    images = images.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(img_size), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
def plot_intervened_images(images, alphas,num_images=10,img_size=(28, 28),save_path=None):
    """
    Plot generated images.
    Args:
        images: Array of intervened images.
        img_size: Size of each image (default: 28x28).
    """
    images = images.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(img_size), cmap='gray')
        
        ax.axis('off')
        ax.set_title(f"alpha={alphas[i]}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
# Function to plot reconstructions and original images
def plot_reconstructions(original, reconstructed, n_images=10, save_path=None):
    """
    Plots a grid of original and reconstructed images for comparison.
    Args:
        original (torch.Tensor): The original images.
        reconstructed (torch.Tensor): The reconstructed images.
        n_images (int): Number of images to display (default: 10).
        save_path (str, optional): Path to save the plot (default: None).
    """
    fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(n_images * 1.5, 3))
    
    for i in range(n_images):
        axes[0, i].imshow(original[i].cpu().numpy().transpose(1, 2, 0))
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].transpose(1, 2, 0))
        axes[1, i].axis('off')

    if save_path:
        
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()





