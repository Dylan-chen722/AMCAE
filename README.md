# Label-efficient deep learning empowered smart control for robust and generalizable decarbonization in energy systems

Related article ***"Label-efficient deep learning empowered smart control for robust and generalizable decarbonization in energy systems"*** is submitted to *Nature Communications*.

## Overview
**The explosive emergence of deep learning has inaugurated a new era in energy systems, with a focus on control optimization to facilitate the process of decarbonization. However, acquiring sufficient labels to train a generalizable deep learning model poses a significant challenge due to the considerable cost, time and labor involved. Here, we developed a novel label-efficient image recognition model named asymmetrically masked convolutional autoencoder (AMCAE) for control optimization in energy systems. Extensive comparative tests demonstrated the superior performance of AMCAE over existing advanced algorithms in terms of both accuracy and generalization. Based on the field experiment conducted in real energy system, the energy saving and carbon emission were anticipated to exceed 196.49 GWH and 90.82 kt on a global scale, respectively. Our study is committed to promoting the practical application of deep learning techniques, with the ultimate aim of accelerating the achievement of net-zero emissions in energy systems.**.

## Hardware and Software Requirements
The project is developed on Windows 10 operating systems. We have tested the program based on the computer with the following specs: Intel Core i9-10900K central processing unit, Nvidia GeForce RTX 3080 graphics processing unit, 32-GB random access memory and 512-GB solid state drive.
The program depends on the Python scientific stack, and the versions used for testing are:

```
Python==3.7.12
numpy==1.21.3
Pillow==8.3.2
torch==1.10.0
torchvision==0.11.1
```
## Code Structure
**The project is structured as follows**:<br>
├── models/<br>
│ ├── AutoEncoder.py<br>
│ ├── ImageClassifier.py<br>
├── scripts/<br>
│ ├── train_CAE.py<br>
│ ├── train_Classifier.py<br>
├── Tools/<br>
│ ├── data_loader.py<br>
│ ├── logger.py<br>
│ ├── mask.py<br>
│ ├── utils.py<br>
├── dataset/<br>
│ ├── train_data<br>
│ ├── test_data<br>
├── results/<br>
├── logs/<br>
├── main_autoencoder.py<br>
├── main_classifier.py<br>
├── main_reconstruction.py<br>
├── requirements.txt<br>
├── README.md<br>
- `models/` contains the autoencoder model and image classification model.
- `scripts/` contains the training scripts of the autoencoder model and image classification model.
- `Tools/` contains utility functions used by the project.
- `Tools/data_loader.py` is the methods for load image and dataset.
- `Tools/logger.py` is the methods for logging the parameters and the results.
- `Tools/mask.py` contains the asymmetrical mask methods used in this project.
- `Tools/utils.py` contains some useful functions for model training.
- `dataset/` contains the training data and the testing data.
- `results/` is used to save the reconstructed image and the residual image.
- `logs/` is used to save the recorded files or contents in the training process.
- `main_autoencoder.py` is the main function for training the autoencoder model.
- `main_classifier.py` is the main function for training the image classification model.
- `main_reconstruction.py` is the main function for obtaining the reconstructed image and residual image.
- `requirements.txt` is the file that contains the dependent libraries required.
- `README.md` is this file that you're reading right now.
## Data Availability

The raw data are protected and are not available due to data privacy laws and commercial interests. Investigators with an academic affiliation may contact the corresponding author for data access for the purposes of validating the above findings. Requests will be processed within 60 days.

## License
This project is covered under the **MIT License**.

## Issues
If you find any bugs or have problems when you are using our program, feel free to raise issues.

