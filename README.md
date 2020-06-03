# HUSE: Hierarchical Universal Semantic Embeddings

![alt text](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/architecture.jpg)

Implementation of HUSE: Hierarchical Universal Semantic Embeddings ( https://arxiv.org/pdf/1911.05978.pdf )

This paper proposes a novel method, HUSE, to learn cross-modal representation with semantic information. HUSE learns a shared latent space where the distance between any two universal embeddings is similar to the distance between their corresponding class embeddings in the semantic embedding space. HUSE also uses a classification objective with a shared classification layer to make sure that the image and text embeddings are in the same shared latent space

HUSE implementation architecture is divided into 3 primary parts:

### PART1: CREATING TEXT AND IMAGE EMBEDDINGS INPUTS:
HUSE being a Multimodal Model takes in two input, image and text. The Image is passed onto a pre-trained Graph-Regularized Image Semantic Embeddings (Graph-RISE) Model which produces an embedding for an individual images and BERT embeddings are used to obtain a representation of the Text.

### PART2: MODEL IMPLEMENTATION FOR CREATING FINAL EMBEDDINGS:
The output from Graph-RISE is passed onto an Image Tower in parallel to output from BERT which is passed onto the Text Tower. The L2 normalized output from both the towers are further passed onto a shared fully connected layer. The output of the shared fully connected layer is further used to calculate different losses.

### PART3: INCORPORATING THREE  LOSSES INTO THE ARCHITECTURE:
The paper incorporates three losses, for Class Level Similarity, Semantic Similarity, Cross Modal Gap. All three losses are explained in detail in the paper. Implementing these three losses is the main objective and thus carries the highest points.




### DATASET

50k products accompanied along with its image, text and class name.

### IMAGE EMBEDDINGS INPUT MODEL

Images are represented to vector embedding space using the img2vec pretrained library. This library uses the ResNet50 model in TensorFlow Keras, pre-trained on Imagenet, to generate image embeddings.

### TEXT EMBEDDINGS INPUT MODEL 

We used BERT pretrained text model for representing text into embedding space. Text embedding created from this part of the model are forwarded into the text tower.

### UNIVERSAL SENTENCE ENCODER

We used universal sentence encoder to make embeddings of the classes.

### IMAGE AND TEXT TOWER MODEL

This part of the model is build as it is taking input from image and text pretrained model embeddings and parameters used are mentioned in the HUSE paper. The image tower consists of 5 hidden layers of 512 hidden
units each and text tower consists of 2 hidden layers of 512 hidden units each with RELU non-linearity and dropout of 0.15 is used between all hidden layers of both towers and the resulting embedding are L2 normalized.

Model passes emmbeddings from image tower and text
tower through a shared fully connected layer and the model is trained using softmax cross entropy loss.

### INCORPORATING THREE LOSES INTO THE MODEL

### CROSS MODEL GAP LOSS

This calculates the cross modal loss, it returns the cosine distance between the image and text embeddings of the same class.

![alt text width = 400](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/cross-model-gap.jpg)

### SEMENTIC SIMILIARITY LOSS

This calculates the semantic loss, it basically calculates the distance between the image embeddings and text embeddings of different classes
and then they are compared to the distance between the respective class embeddings and the loss is calculated 

![alt text width = 400](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/sementic-similiarity-loss.jpg)

### CLASS LEVEL SIMILIARITY LOSS

This calculates the classification loss, we use simple categorical crossentropy loss for this.

![alt text width = 400](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/class-level-similarity.jpg)

### OVERALL LOSS

![alt text width = 400](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/overall-loss.jpg)

