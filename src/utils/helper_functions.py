import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax



def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')



def save_tensor_image(
    image, 
    save_dir='.', 
    filename=None, 
    normalize=True
):
    """
    Save a PyTorch tensor as an image file and return its path.

    Args:
        image (Tensor):       3×H×W tensor (C×H×W).
        save_dir (str):       Directory where to save the image.
        filename (str|None):  If provided, use this name; otherwise generate a UUID.
        normalize (bool):     If True, un‐normalize from ImageNet stats.

    Returns:
        str:  The full path to the saved image file.
    """
    # ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # generate a filename if needed
    if filename is None:
        filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(save_dir, filename)

    # move to CPU & convert to H×W×C numpy
    arr = image.cpu().numpy().transpose(1, 2, 0)

    # un‐normalize (if desired)
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr = std * arr + mean
        arr = np.clip(arr, 0, 1)

    # save out as a PNG
    plt.imsave(path, arr)

    return path
