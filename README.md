# CNN Image Classification

Built two convolutional neural networks from scratch to classify images. One for handwritten digits (MNIST), another for real-world objects (CIFAR-10).

## What it does

Trains deep learning models that can recognize:
- **MNIST**: Handwritten digits (0-9)
- **CIFAR-10**: Airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks

## Results

- MNIST: ~99% accuracy
- CIFAR-10: ~85% accuracy

Not bad for 30 epochs of training.

## Tech Stack

```
TensorFlow 2.x
Keras
NumPy
Matplotlib
```

## Quick Start

```bash
# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook cnn_image_classification.ipynb
```

Then run all cells. Training takes about 15-20 minutes on GPU, longer on CPU.

## Architecture

### MNIST Model
```
Conv2D (32) → BatchNorm → Conv2D (32) → BatchNorm → MaxPool → Dropout
Conv2D (64) → BatchNorm → Conv2D (64) → BatchNorm → MaxPool → Dropout
Dense (128) → BatchNorm → Dropout → Dense (10)

Parameters: ~200K
```

### CIFAR-10 Model
```
Conv2D (32) → BatchNorm → Conv2D (32) → BatchNorm → MaxPool → Dropout
Conv2D (64) → BatchNorm → Conv2D (64) → BatchNorm → MaxPool → Dropout
Conv2D (128) → BatchNorm → Conv2D (128) → BatchNorm → MaxPool → Dropout
Dense (256) → BatchNorm → Dropout → Dense (10)

Parameters: ~750K
```

## Key Features

**Data Augmentation**
- Random rotations (10-15°)
- Width/height shifts
- Horizontal flips (CIFAR-10 only)
- Zoom

**Training Tricks**
- Batch normalization (stabilizes training)
- Dropout (prevents overfitting)
- Early stopping (stops when not improving)
- Learning rate reduction (adjusts LR automatically)
- Model checkpointing (saves best model)

**Visualizations**
- Training/validation curves
- Confusion matrices
- Sample predictions with correct/incorrect labels
- Augmentation examples

## What I learned

Building this taught me:
- How CNNs extract features (edges → shapes → objects)
- Why pooling layers matter (reduce dimensions, prevent overfitting)
- Batch normalization makes training way more stable
- Data augmentation is crucial for small datasets
- CIFAR-10 is WAY harder than MNIST (real images vs clean digits)

Debugging took longer than expected. Overfitting was a problem until I added more dropout and augmentation.

## Training Details

### MNIST
- Epochs: 30
- Batch size: 128
- Learning rate: 0.001
- Time: ~5 minutes

### CIFAR-10
- Epochs: 30
- Batch size: 64
- Learning rate: 0.001
- Time: ~15 minutes

Both use Adam optimizer and categorical cross-entropy loss.

## Files

```
cnn_image_classification.ipynb    # Main notebook
README.md                          # This file
mnist_cnn_best.h5                 # Saved MNIST model
cifar_cnn_best.h5                 # Saved CIFAR-10 model
```

## Improvements to try

**Architecture changes:**
- Add residual connections (ResNet style)
- Try depthwise separable convolutions
- More layers (go deeper)

**Training improvements:**
- Different optimizers (AdamW, RAdam)
- Cosine annealing for learning rate
- Mixup or Cutmix augmentation
- Label smoothing

**Transfer learning:**
- Use pre-trained models (ResNet50, EfficientNet)
- Fine-tune on these datasets
- Should hit 95%+ on CIFAR-10

**Ensemble:**
- Train 3-5 models with different seeds
- Average their predictions
- Usually adds 2-3% accuracy

## Common Issues

**Out of memory**
- Reduce batch size to 32 or 16
- Use smaller model (fewer filters)

**Slow training**
- Check if GPU is being used: `tf.config.list_physical_devices('GPU')`
- If no GPU, reduce epochs or use smaller model

**Poor accuracy**
- Train longer (increase epochs)
- Adjust learning rate
- Add more augmentation
- Check if data is normalized properly

**Overfitting**
- Increase dropout rates
- Add more augmentation
- Use early stopping (already included)

## Requirements

```txt
tensorflow>=2.10.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

## Dataset Info

**MNIST**
- 60,000 training images
- 10,000 test images
- 28x28 grayscale
- Already pretty clean

**CIFAR-10**
- 50,000 training images
- 10,000 test images
- 32x32 RGB
- Real-world photos (harder)

Both download automatically when you run the code.

## Next Steps

Planning to try:
1. Transfer learning with ResNet
2. Test on custom images
3. Add more visualizations (filter activations)
4. Try on different datasets (CIFAR-100, Fashion-MNIST)

Built to understand CNNs properly instead of just using pre-trained models.
