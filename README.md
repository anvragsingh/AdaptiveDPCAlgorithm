# AdaptiveDPCAlgorithm

This project implements Adaptive DPC, a training algorithm that dynamically adjusts quantization precision for each layer in a neural network to optimize computational efficiency without sacrificing accuracy.

Adaptive DPC solves this by:
<br>
-Dynamically adjusting bit-widths during training based on gradient statistics
<br>
-Assigning lower precision to higher-capacity layers
<br>
-Using quantization as a regularization method



├── AdaptiveDPCAlgorithm.py                      # Core logic for adaptive precision assignment
<br>
├── QuantizedResNet.py                             # Custom quantized layers and ResNet integration
<br>
├── train.py                                   # Training script with CLI for config
<br>
├── rainingLoopwithAdaptiveDPC.py                       # Training loop with ADPC integration
<br>
├── utils.py                                          # Dataset loading, evaluation, and utility functions
<br>
├── Adaptive DPC.pptx                            # Project presentation
<br>
├── research paper summary.pdf                # Literature summary 
<br>
├── technical documentation.pdf              # Additional documentation
<br>
<br>
<h3>How It Works</h3>

-Initialization:
<br>
--Each layer is assigned an initial bit-width based on its capacity using a logarithmic schedule.

-Quantization:
<br>
--During each forward pass, weights are quantized before computation.

-Gradient Monitoring:
<br>
--After backpropagation, gradient statistics are collected.

-Bit-width Update:
<br>
--Layers with stable gradients reduce precision.
<br>
--Layers with noisy gradients increase precision.

<h3>Future Work</h3>
-Quantization of activations and gradients
<br>
-Support for transformer-based models
<br>
-Deployment on edge devices in real-time
