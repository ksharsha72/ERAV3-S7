# CNN Model Architecture

This repository contains a Convolutional Neural Network (CNN) implementation designed for image classification tasks. The model is particularly structured for MNIST-like datasets (28x28 input images).

## Architecture Overview

The network consists of three main blocks:
1. Initial Feature Extraction (Conv1-3)
2. Dimensionality Reduction (Pool + Antenna)
3. Final Feature Processing (Conv4-7)

### Layer-by-Layer Breakdown 

                                     Model Architecture
                                     ==================

                                Input Image (1×28×28)
                                        ↓
                                ┌───────────────────┐
                                │ Conv1 (3×3)       │
                                │ Output: 8×26×26   │──→ ReLU ──→ Dropout ──→ BatchNorm
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ Conv2 (3×3)       │
                                │ Output: 12×24×24  │──→ ReLU ──→ Dropout ──→ BatchNorm
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ Conv3 (3×3)       │
                                │ Output: 16×22×22  │──→ ReLU ──→ Dropout ──→ BatchNorm
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ MaxPool2d (2×2)   │
                                │ Output: 16×11×11  │
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ Antenna Conv (1×1)│
                                │ Output: 8×11×11   │──→ ReLU ──→ Dropout ──→ BatchNorm
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ Conv4 (3×3)       │
                                │ Output: 10×9×9    │──→ ReLU ──→ Dropout ──→ BatchNorm
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ Conv5 (3×3)       │
                                │ Output: 14×7×7    │──→ ReLU ──→ Dropout ──→ BatchNorm
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ Conv6 (3×3)       │
                                │ Output: 16×5×5    │──→ ReLU ──→ Dropout ──→ BatchNorm
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ AvgPool2d (5×5)   │
                                │ Output: 16×1×1    │
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │ Conv7 (1×1)       │
                                │ Output: 10×1×1    │
                                └───────────────────┘
                                        ↓
                                ┌───────────────────┐
                                │    LogSoftmax     │
                                │ Output: 10 classes│
                                └───────────────────┘

### Network Details

#### Block 1: Initial Feature Extraction
- Starts with 1 channel (grayscale input)
- Progressively expands channels: 8 → 12 → 16
- Each conv layer has: Conv2d → ReLU → Dropout(0.05) → BatchNorm
- Reduces spatial dimensions: 28×28 → 26×26 → 24×24 → 22×22

#### Block 2: Dimensionality Reduction
- MaxPool2d reduces spatial dimensions by half: 22×22 → 11×11
- Antenna (1×1 conv) reduces channels: 16 → 8
- Maintains spatial dimensions while reducing complexity

#### Block 3: Final Processing
- Further feature processing: channels 8 → 10 → 14 → 16
- Spatial reduction: 11×11 → 9×9 → 7×7 → 5×5
- Global average pooling: 5×5 → 1×1
- Final 1×1 conv for classification: 16 → 10 classes

#### Key Features
- No fully connected layers (fully convolutional)
- Consistent dropout (0.05) throughout
- BatchNorm after every conv layer
- Mix of 3×3 and 1×1 convolutions
- Strategic use of pooling layers

`
The above model is developed in Three phases.
Phase 1: Model Development
         a lighter model is considered with around 7042 parameters.in S7_1.ipynb file.
         target accuracy is 99.4
         Result:
                train accuracy: 99.0
                test accuracy: 98.72
                parameters: 7042
                Analysis:
                        The model is has less gap between train and test accuracies
                        The model is not able to generalize the data.
                        The model can be furture imporoved through regularization, BatchNormalxaion and Image augmentation

Phase2: Model is considered with around 7210 parameters.in S7_2.ipynb file.
         target accuracy is 99.4
         Result:
                train accuracy: 99.10
                test accuracy: 99.35
                parameters: 7210
                Analysis:
                        After adding dropout 0.05 the model performance is improved.
                        The model is clearly underfitting the data. and gap is small
                        The model training accuracy can be increased by introducing the image augmentation and introducing the scheduler

Phase3: Model is considered with around same 7210 parameters.in S7_3.ipynb file.
        since no parameters were introduced, the model is same as phase 2.but the model is improved by adding the image augmentation and scheduler. and playing with the learning rate.


        target accuracy is 99.4
        results:
                train_accurcay: 98.97
                test_accuarcy: 99.43
                paramters: 7210
                Analysis, the image augmentation with a learning rate of 0.03 has certainly improved accuarcy and was able to acheive 99.4 twice and the modle is underfitting

`




