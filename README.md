# BrainTumorSegmentation
Perform segmentation on MRI images of Brain Tumor using deep learning model, Unet

# Introduction
When caught early and treated properly, a brain tumor diagnosis does not have to be life-threatening. But with so many types of brain tumors, accurately diagnosing one can be a complicated process. Brain MRI provides detailed images often used by doctors to determine the location and type of tumor. Extreme precision is always required in both biopsy and surgical resection. Imaging also important during follow-up to monitor tumor growth or recurrence and detect new tumor cells in the brain. Therefore ability to map out the tumor precisely can be invaluable in evaluation pre- and post-treatment. 

However manual segmentation of brain tumor from 3D MRI volumes is a very time-consuming task and the performance is highly relied on operator’s experience. Moreover, reproducible results are difficult to achieve even by the same operator.

In recent years, automatic segmentation based on deep learning methods has been widely used, where a neural network can automatically learn image features. In medical image analysis field, U-net is one of the most important architecture to carry out medical semantic segmentation task. It can accurately segment the desired feature target such as tumors' size, shape, regularity, location and their heterogenous appearance. Implementing U-Net for brain tumor segmentation from MRI images might be useful in improving its diagnosis and management.

# Problem Statement
Brain tumors are a heterogeneous group of central nervous system neoplasms that arise within or adjacent to the brain. The location of the tumor within the brain has a profound effect on the patient's symptoms, surgical therapeutic options, and the likelihood of obtaining a definitive diagnosis. The location of the tumor in the brain also markedly alters the risk of neurological toxicities that alter the patient's quality of life.  Traditionally, brain tumors are detected by imaging only after the onset of neurological symptoms.  

Current imaging techniques provide meticulous anatomical delineation and are the principal tools for establishing that neurological symptoms are the consequence of a brain tumor. There are many techniques for brain tumor detection. We have used semantic segmentation using the UNET Architecture for brain tumor detection to solve this task with better efficiency and accuracy.  

UNET was our preferred choice of architecture because its proven to be more successful conventional models in terms of architecture and in terms pixel-based image segmentation formed from convolutional neural network layers. It’s even effective with limited dataset images. At convolutional layers, we used RELU activation function in common with encoder and decoder. In the encoder, max pooling is used for downsampling. In the decoder, deconvolution is used for upsampling. The most important characteristic of UNET is skip connection between encoder and decoder. The feature map with the position information in the encoder is concatenated to the restored feature map in the decoder. Therefore, the position information is complemented, and each pixel can be more accurately assigned to the detected tumor. The presentation of this architecture was first realized through the analysis of biomedical images and its well suited for our application.

#Idea of project
The main idea is to do semantic segmentation on brain tumor detection. In this project, we developed a fully convolutional network that is based on U-Net architecture. A comprehensive data augmentation technique has been used to increase the segmentation accuracy, overall model performance and mean IOU. This is because after every concatenation in U-Net architecture, again apply consecutive regular convolutions so that the model can learn to assemble a more precise output. In addition, we applied a cross entropy as a loss function and sigmoid as activation. Due to unbalance samples of brain tumor regions, we need to trial and error by changing parameters and hyperparameters to achieve good result such as learning rate, number of epochs, weight initialization, activation function, loss function and batch size. The proposed method has been validated using datasets acquired for both training and test set. 3379 and 550 images had been used for training and test set respectively.

#The Journey of the project 
##1)	Data Collection 
 
         As we have decided on the problem to be solved, we began to search for a suitable dataset to train and test the model. Numerous datasets were reviewed to study the suitability of the dataset towards our project. We decided to utilize a brain tumor dataset (‘Br35H :: Brain Tumor Detection 2020), uploaded by Ahmed Hamada in Kaggle. The downloaded dataset contained 3650 images, which we further sorted out into two categories, which are input images and their corresponding masked images. This is to ensure the trained model could be able to detect segmentation effectively. On the initially stage of training the model, the system faced errors where it couldn’t manage find the masked image. To solve this issue, we renamed all the masked images similarly to their corresponding MRI images. Once the sorting and renaming part was done, various training and testing sizes were implemented to acquire the best model.  
 
##2)	Architecture 
         As for the architectural model, we decided go upon UNET, a convolutional network architecture for fast and precise segmentation of images. UNET architecture also aids transposed convolution, perform up sampling of an image with learnable parameters. This basically means that the module will be able to receive a low-resolution image and process it to be a high-resolution image. We have also implemented transfer learning where we imported a pre-trained UNET model to act as the feature extractor for the input images. Additionally, a CNN Loss Layer was added to establish an output layer and to fully form a fully convolutional network.  

 
##3)	Modelling 
                Various set of hyperparameters such as batch size, activation function and loss functions were tested to identify under which hyperparameters, the module works the best. The efficiency level of the hyperparameters were measured by the acquired accuracy, precision, recall, F1 score and the mean IOU. Apart from that, the graph was also observed to study the effectiveness of the implemented hyperparameters in decreasing the loss as the training goes on. Upon numerous trials and errors, we finalized on the below hyperparameters to build our best version of the module.  

HYPERPARAMETER 	                 DETERMINED VALUE 
Learning rate 	                     3e-3 
Decay Rate 	                          0.5 
Weight Initiator 	                   Xavier 
Loss Function 	                   Cross Entropy 
Activation Function 	              Sigmoid 
Step Size 	                           5 
Dimension of the Input Images      	224*224*1 
Color of the Input Images 	        Grayscale 
 
As for the parameters, we determined the input data to be 3379, with a split of 80 percent for training and 20 percent for testing. The module was set to train with the seed value of 1234, 100 epochs and at the batch size of 5. 

##4)	Evaluation
      From the evaluation of the trained UNET model was using 550 brain tumor MRI images in the test data set, it has found that the mean Intersection of Union (IOU) detected by the model was 0.685. It showed there was a slightly dropped of performance from the IOU obtained when training the UNET model was 0.779. This showed that in detection of new MRI images, there are some variation in terms of pixels of tumor in the MRI images in the test data that the model not able to fully detect and well segmented. In terms of evaluation matrix, the trained UNET model has the accuracy 0.9925, precision 0.9173, Recall 0.8996 and F1 score 0.9083. These figures proved that the UNET model was generalizable and having reasonable prediction and segmentation on the new brain tumor images.  

##5)	Possible Improvement
    The UNET model could be further improved by running the training at much higher epochs to increase the training time the model recognized and assigned pixel to pixel segmentation more accurately. In regards to prevent overfitting problem of the UNET model, finding more sources of MRI images and feed to the training to enable UNET able to learn the variation in Brain tumor MRI images in terms of shape, size, location so it could have more accurate prediction and segmentation. It was also suggested to modify the parameters of the frozen layer so that it could have more hyperparameters tuning, this is because sometimes, the domain of images of the pretrained model may have large different compared to the domain of images of the raw input images, so to increase the model prediction capability, having more fine-tune layer to training more parameters is a solution. Lastly, it also interested to adopt arbiter in deeplearning4j for optimization and hyperparameters searching to obtain the best parameter for the UNET in semantic segmentation of brain tumor in the MRI images. 
