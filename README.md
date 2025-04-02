# IMAGE-CLASSIFICATION-MODEL

**COMPANY**: CODTECH IT SOLUTIONS

**NAME**: GALI SUPRIYA

**INTERN ID**:CT12WJVC

**DOMAIN**: MACHINE LEARNING

**BATCH DURATION**:JANUARY 5TH TO APRIL 5TH,2025

**MENTOR NAME**:NEELA SANTHOSH 

# Face Mask Detection Using CNN

Face mask detection is a crucial application of deep learning, especially in public health monitoring and security systems. This project implements a *Convolutional Neural Network (CNN)* to classify images into three categories: *"With Mask," "Without Mask," and "Mask Worn Incorrectly."* The model is trained using *TensorFlow* and evaluates its performance using classification metrics.

## Tools and Technologies Used
- *Programming Language:* Python
- *Deep Learning Framework:* TensorFlow/Keras
- *Libraries:* NumPy, OpenCV, Matplotlib, Seaborn, scikit-learn
- *Dataset Handling:* Custom dataset of face images with labeled categories
- *Hardware:* Can run on CPU but performs better on GPU

## Dataset and Preprocessing
The dataset consists of images categorized into three labels. Preprocessing steps include:
1. *Loading Images:* Using OpenCV, images are loaded and resized to *64x64 pixels*.
2. *Normalization:* Pixel values are scaled between *0 and 1* for better training stability.
3. *Label Encoding:* Labels are converted into integers (*0, 1, 2*) for categorical classification.
4. *Data Splitting:* The dataset is divided into *80% training* *10% validation*and *10% testing* for model evaluation.

## Model Architecture
The CNN consists of:
1. *Convolutional Layers (Conv2D):* Extract spatial features from images.
2. *MaxPooling Layers:* Reduce feature map size and computational cost.
3. *Flatten Layer:* Converts 2D features into a 1D array.
4. *Fully Connected Layers:* Process extracted features for final classification.
5. *Softmax Activation:* Outputs probabilities for each category.

## Training and Optimization
The model is compiled using *Adam optimizer* with *sparse categorical cross-entropy loss* for multi-class classification. Training occurs over *10 epochs* using *training data for learning* and *validation data for tuning model performance*.

## Model Performance and Evaluation
After training, the model achieves an accuracy between * 98% *. The evaluation metrics include:
1. *Classification Report:* Displays precision, recall, and F1-score for each category.
2. *Confusion Matrix:* Shows model prediction accuracy for each label using Seaborn visualization.

## Output and Observations
- The model successfully classifies most images correctly, with some misclassification due to *similar facial features* or *mask positioning issues*.
- The confusion matrix highlights areas where the model struggles, particularly between *"With Mask" and "Mask Worn Incorrectly."*
- Increasing dataset size and adding *data augmentation* can further improve accuracy.

## Applications
1. *Public Health and Safety:* Automated mask detection in public spaces, ensuring compliance with regulations.
2. *Security Systems:* Integrating with *CCTV surveillance* to monitor mask usage in restricted areas.
3. *Retail and Workplaces:* Ensuring staff and customer adherence to mask-wearing policies.
4. *Smartphone Applications:* Real-time mask detection using mobile cameras for personal safety checks.
5. *AI-Powered Access Control:* Allowing or restricting entry based on face mask compliance in offices, malls, or hospitals.

## Classification metrics 

![Image](https://github.com/user-attachments/assets/43618980-1fcf-42e1-9bd7-372efdb73804)

## Confusion matrix 

![Image](https://github.com/user-attachments/assets/aacba551-8501-4134-87db-c146bffcaade)

## Accuracy and loss plots 

![Image](https://github.com/user-attachments/assets/de0be4b9-3a9e-49aa-9b12-6f06e7695d00)

## Predicted output

![Image](https://github.com/user-attachments/assets/bf3c6a7c-7477-4db0-8574-430036b0e9c4)

![Image](https://github.com/user-attachments/assets/2dc48aa7-9ef1-4aea-87d8-b4a3ae1983a9)

![Image](https://github.com/user-attachments/assets/28e4e98f-0a1a-48ad-b764-81f02b2432e2)
