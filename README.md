# msot_muscle_segmenation

This project aims at the Segmentation of Muscle Tissue in Multispectral Optoacoustic Tomography Images.

For this purpose, a Deep Learning architecture will be used. The baseline  behavior is established on a classic UNET, which is then slightly altered to be used for an active deep learning framework, namely include Monte Carlo Dropout techniques for uncertainty estimation.

The network's pipeline has been designed as follow: Initially a classic UNET, with some dropout layers introduced is trained on the data and the segmentation mask and the uncertainty mapping are being predicted. Subsequently, a second CNN has been implemented for the quality estimation of the network, namely predict the Dice score and compare how well the network is performing.

Evaluation metrics:

The Dice Coefficient metric is being used for evaluation of UNET and the mean squared error (mse) for the vgg-like CNN.
