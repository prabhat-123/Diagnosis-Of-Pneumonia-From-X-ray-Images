# Diagnosis Of Pneumonia From X-ray Images

We have seen tremendous research going on in medical imaging by utilizing available medical CT scan images and xray images.
On November 15, 2017, Stanford researchers have developed an algorithm that offers diagnoses based on chest X-ray images.
It can diagnose up to 14 types of medical conditions and is able to diagnose pneumonia better than expert radiologists working alone.
Thus, it is possible to use Deep Learning algorithms to detect the disease from images of Chest X-rays and CT scans.
Automated applications can be created to help support radiologists.
So,this project aims to detect pneumonia cases from the x-ray images.
This project will be an end to end solution where the user/radiologist will upload the chest x-ray image to the app and the app would return the predictions denoting whether the patient has pneumonia or not.

**1) Problem Statement**

<li>According to the latest WHO data published in 2018 Influenza and Pneumonia Deaths in Nepal reached 9,712 or 5.83% of total deaths.</li>
<li>Chest pain, Difficulty in breathing, Respiratory problems - Symptoms of Pneumonia</li>
<li>All these respiratory symptoms can be identified by the radiologist by observing the chest x-ray images of the patient.</li>

<br>

**2) Aims & Objectives**
<li> So, the objective of the project is to create a deep learning algorithm to identify pneumonia patients by interpreting chest x-ray images.</li>
<li>This project will be an end to end solution where the user/radiologist will upload the chest x-ray image to the app and the app would return the predictions denoting whether the patient has pneumonia or not.</li>

<br>

**3) Data Exploration**
<li>There are a total of 5856 image datasets.</li>
<li>Anterior-posterior Chest X-ray images were selected from the patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.</li>
<li>The overall image datasets are categorized into two main classes : 'NORMAL' and 'PNEUMONIA'.</li>
<li>The overall dataset is divided into three different splits : training, testing and validation.</li>

<br>

**Train Test Split Ratio**

![](output_images/split_representation.png)

<br>

**Distribution Of Each Class In the Overall Dataset**

![](output_images/class_representaion.png)

<br>

**4. Data Preprocessing : Image Processing**

*****a)Image Resizing*****
  <li>The image is resized into (400, 400) pixels.</li>

<br>

*****b)Scaling pixel values*****
  <li>The pixel values of an image is scaled by dividing the each pixel position by 255.</li>

<br>

**5. Modelling**

<li>Selecting the pretrained model : Inception V3 (Google Net)</li>
<li>Using Transfer Learning technique to train the network faster.</li>
<li>Selecting Hyper Parameters (Learning Rate, Batch Size)</li>
<li>Loss Functions (Weighted Binary Crossentropy)</li>
<li>Optimizer (Adam)</li>

<br>

**Training Vs Validation Accuracy**

![](output_images/training_vs_validation_accuracy_pneumonia_detection.jpg)

<br>

**Training Vs Validation Loss**

![](output_images/training_vs_validation_loss_pneumonia_detection.jpg)



<br>

**Overall System Workflow**

![](output_images/rsz_high_level_diagram.png)
