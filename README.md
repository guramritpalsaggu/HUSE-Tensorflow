# HUSE: Hierarchical Universal Semantic Embeddings
Implementation of HUSE: Hierarchical Universal Semantic Embeddings ( https://arxiv.org/pdf/1911.05978.pdf )

![alt text](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/architecture.jpg)


### Dataset

50k products accompanied along with its image, text and class name.

#### Image Embeddings Input Model

Images are represented to vector embedding space using the img2vec pretrained library. This library uses the ResNet50 model in TensorFlow Keras, pre-trained on Imagenet, to generate image embeddings.

#### Text  Embeddings Input Model 

We used BERT pretrained text model for representing text into embedding space. Text embedding created from this part of the model are forwarded into the text tower.

#### Universal Sentence Encoder

We used universal sentence encoder to make embeddings of the classes.

#### Image Tower Model and Text Tower Model. 

This part of the model is build as it is taking input from image and text pretrained model embeddings and parameters used are mentioned in the HUSE paper. The image tower consists of 5 hidden layers of 512 hidden
units each and text tower consists of 2 hidden layers of 512 hidden units each with RELU non-linearity and dropout of 0.15 is used between all hidden layers of both towers and the resulting embedding are L2 normalized.

Model passes emmbeddings from image tower and text
tower through a shared fully connected layer and the model is trained using softmax cross entropy loss.

#### Incorporating three losses into the model architecture

#### Cross Model Gap Loss

This calculates the cross modal loss, it returns the cosine distance between the image and text embeddings of the same class.

![alt text width = 400](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/cross-model-gap.jpg)

#### Sementic Similarity Loss

This calculates the semantic loss, it basically calculates the distance between the image embeddings and text embeddings of different classes
and then they are compared to the distance between the respective class embeddings and the loss is calculated 

![alt text](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/sementic-similiarity-loss.jpg)

#### Class Level Similarity Loss

This calculates the classification loss, we use simple categorical crossentropy loss for this.

![alt text](https://github.com/guramritpalsaggu/HUSE-Tensorflow/blob/master/resources/class-level-similarity.jpg)

