import torch
import torchvision
from torchvision import transforms as T

def load_and_preprocess_data(dataset_path='./data', batch_size=64, img_size=(28, 28), sup_frac=0.5):
    """
    Load and preprocess the dataset, supporting semi-supervised settings.
    Args:
        dataset_path: Path to the dataset.
        batch_size: Size of each batch of data.
        img_size: Target image size for resizing.
        sup_frac: Fraction of labeled data to use for supervised training (0.0 to 1.0).
    Returns:
        loaders: Dictionary containing DataLoader objects for supervised, unsupervised, validation, and testing.
    """
    # Prepare data transformations 
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
        lambda x: (x > 0.5).float() # Binarize the data
    ])
    
    # Load full training and test data
    full_training_data = torchvision.datasets.MNIST(dataset_path, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(dataset_path, train=False, transform=transform, download=True)
    
    # Create train and validation splits
    num_samples = len(full_training_data)
    training_samples = int(num_samples * 0.8)  # 80% training
    validation_samples = num_samples - training_samples  # 20% validation
    training_data, validation_data = torch.utils.data.random_split(
        full_training_data, [training_samples, validation_samples]
    )
    
    # Supervised and Unsupervised Splits
    if sup_frac > 0:
        sup_samples = int(len(training_data) * sup_frac)
        unsup_samples = len(training_data) - sup_samples
        sup_data, unsup_data = torch.utils.data.random_split(
            training_data, [sup_samples, unsup_samples]
        )
    else:
        raise ValueError("sup_frac must be greater than 0 for semi-supervised settings.")
    
    # Initialize dataloaders
    loaders = {
        'sup': torch.utils.data.DataLoader(sup_data, batch_size=batch_size, shuffle=True),
        'unsup': torch.utils.data.DataLoader(unsup_data, batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False),
        'test': torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False),
    }
    
    return loaders