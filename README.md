# Deep Convolutional GAN (DCGAN) --- Face Image Generation

A PyTorch implementation of a Deep Convolutional Generative Adversarial
Network (DCGAN) trained on face images from the CelebA dataset. The
generator learns to create **64 × 64 RGB face images** from random
100-dimensional latent vectors, while the discriminator learns to
distinguish real training images from generated images.

## Highlights

-   Built a generator with fully connected projection, batch
    normalization, transposed convolutions, ReLU activations, and a
    final `Tanh` activation.
-   Built a convolutional discriminator with LeakyReLU activations,
    dropout, and a sigmoid output for real/fake classification.
-   Trained the generator and discriminator adversarially using binary
    cross-entropy loss and the NAdam optimizer.
-   Loaded images from a local folder using a custom PyTorch `Dataset`
    and `DataLoader`.
-   Applied center cropping, resizing to 64 × 64, tensor conversion, and
    normalization to the range `[-1, 1]`.
-   Visualized generated samples during training and generated a batch
    of images after training.

## Model architecture

### Generator

Takes a random latent vector of size 100, projects it to a feature map,
and progressively upsamples it with `ConvTranspose2d` layers to produce
a 3-channel 64 × 64 image.

### Discriminator

Uses strided `Conv2d` layers to reduce the spatial dimensions and
increase feature channels, then outputs a probability that an input
image is real.

## Requirements

-   Python 3
-   PyTorch
-   torchvision
-   Pillow
-   Matplotlib
-   A CUDA-enabled NVIDIA GPU (the notebook places the models and
    training tensors on `cuda`)

Install the Python packages:

``` bash
pip install torch torchvision pillow matplotlib
```

Install a PyTorch build compatible with your CUDA setup by following the
official [PyTorch installation
selector](https://pytorch.org/get-started/locally/).

## Dataset setup

Download the CelebA dataset and extract the aligned face images. The
notebook expects the images to be available in this directory relative
to the notebook:

``` text
./img_align_celeba/img_align_celeba/
```

The folder should contain image files such as `.jpg`, `.jpeg`, or
`.png`. If your images are stored elsewhere, update the `root` argument
passed to `CelebAFolder`.

The notebook: 1. Center-crops each image to 178 pixels. 2. Resizes it to
64 × 64. 3. Converts it to a tensor. 4. Normalizes RGB channels using
mean and standard deviation of `0.5`, mapping pixel values to `[-1, 1]`.
5. Randomly splits the images into 90% training and 10% test subsets.

## Run the notebook

1.  Open `dcgan.ipynb` in Jupyter Notebook or VS Code.
2.  Ensure the CelebA image folder is in the expected location, or
    update its path.
3.  Select a Python environment with PyTorch and the required packages
    installed.
4.  Ensure CUDA is available, since the current implementation
    explicitly uses `"cuda"`.
5.  Run the notebook cells in order to define the models, load the
    dataset, train the GAN, and display generated images.

The notebook trains for 20 epochs by default. The generator and
discriminator learning rates are both set to `2e-4`, and the training
batch size is `128`.

## Notes

-   GAN training can be unstable; image quality depends on dataset size,
    training duration, and hyperparameters.
-   The notebook creates a test split and `test_loader`, but the current
    training code does not evaluate the model on that split.
-   Generated images are samples learned from the training distribution,
    not reconstructions of specific input images.
-   No quantitative image-quality metric is calculated in the current
    notebook.

## Repository contents

``` text
.
├── dcgan.ipynb      # Model, data loading, training, and image generation
└── README.md        # Project documentation
```

## Skills demonstrated

PyTorch · Deep Learning · Generative Adversarial Networks · CNNs ·
Transposed Convolutions · GPU Training · Image Preprocessing · CelebA
