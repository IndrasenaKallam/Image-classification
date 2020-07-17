# Classification of Satellite aerial Images using CNN
## Springboard Capstone 2 Project

Satellite imagery is crucial for many applications in agriculture, city planning, natural resource management, environmental monitoring, and disaster response. Since satellite imagery can cover large areas, the labor costs to manually categorize land uses covered by imagery can be prohibitive. Deep learning has emerged as an important approach for automating land use classification over extensive areas.

For this project, I have applied deep learning using convolutional neural networks (CNNs) to classify labelled satellite image tiles from the DeepSat-6 dataset into six land use classes. Two approaches were implemented in Keras: building a custom CNN from scratch and using a pre-trained CNN. Both approaches achieved over 96% accuracy on held-out data.
The custom CNN was also applied to classify new, contiguous satellite imagery not part of the original DeepSat-6 dataset.

This repository includes three final notebooks:
  * Data Acquisition, Checking, and Preparation
  * Satellite Imagery Classification with a Simple CNN
