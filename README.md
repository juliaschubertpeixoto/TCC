# Graduation Work
**Title:** Use of deep learning for gesture recognition through images captured by a kinect sensor in MSRC-12 and NTU RGB+D data basis 

**Abstract:** This work presents an application of deep neural networks for gesture recognition through images captured by a Kinect sensor. In order to perform the training of neural networks, two datasets are used: MSRC-12 and NTU RGB + D. Both datasets consist of a sequence of human body joints movements represented by the skeleton of Microsoft's Kinect sensor. In addition, the FastDTW algorithm is used to normalize the number of data frames. In the MSRC-12 database, three methods of extracting joint characteristics are used: the 3D coordinate method, the normalization method and the subtraction method. In the NTU RGB + D database, only the 3D coordinate method is used. Both datasets were trained in a convolutional neural network model and in a recurrent neural network model. The objective of this work is to verify the assertiveness of the proposed models and joint coordinates features for gesture recognition. The databases were divided into samples of training, validation and testing. The MSRC-12 database showed accuracy greater than 80% using the three methods of extracting joint characteristics in the test sample in the convolutional neural network and in the recurrent neural network. The NTU RGB + D database showed test accuracy greater than 60% in the application using 12 gestures, and greater than 55% in the application using 24 gestures.

# Data
Preprocessed datasets are available [here](https://drive.google.com/drive/folders/19wRvK5Zq7aI9SG-zGZlwbj6ta1db1s7U?usp=sharing).

# Publications
- Peixoto J.S., Pfitscher M., de Souza Leite Cuadros M.A., Welfer D., Gamarra D.F.T. (2021) Comparison of Different Processing Methods of Joint Coordinates Features for Gesture Recognition with a CNN in the MSRC-12 Database. In: Intelligent Systems Design and Applications. ISDA 2020. Advances in Intelligent Systems and Computing, vol 1351. Springer, Cham. https://doi.org/10.1007/978-3-030-71187-0_54
