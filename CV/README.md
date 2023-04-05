# AutoEncoders

  - Recommendation system based on Autoencoder?

## Manifold Learning
One of most important function of Autoencoder is learning 'Manifold'  
Maniford means a sub-space $(\tau)$ that can well explain original data.  
We assume that $p(\tau)$ is smooth, distributed uniformly, and noise is small

### Objectives of Maniford
  - Data Compression
  - Data Visualization
  - Alleviate Curse of Dimensionality
  - Discovering most important features

### Discover of features
  - Euclidian Distance in High dimension might not reflect 'semantic' distance we usually think

## Introduction
Autoencoder (diabolo)
<img width="860" alt="스크린샷 2023-04-05 오후 4 29 39" src="https://user-images.githubusercontent.com/88100984/230215850-af4d68b3-405b-4649-9ce7-c5845284ffe1.png">

Loss encourages output to be close to input
$L(x,y)$

Unsupervised Learning -> Supervised Learning (self-supervised learning)

After traning, encoder and decoder can be used in different context
  - Decoder: at least, able to generate training data
  - Encoder: at least, well represent input data in latent space

### Multi-Layer Perceptron

<img width="1094" alt="스크린샷 2023-04-05 오후 4 35 05" src="https://user-images.githubusercontent.com/88100984/230217149-6df646e1-713b-400d-956e-1caf3d3807f2.png">








## Style GAN
### Style Transfer?
Content + Style  
In Aluminum Bat, Aluminum: style; Bat: Content  
If we transfer the style of Wood Bat,  
expected output will be Wood Bat, while maintaining major characteristics of aluminum bat
### Main Idea
Mapping z to w space so that distribution becomes more linear and thus allows smooth transition.
This alleviates the problem of Gaussian distribution where sampling was done in rather non-linear space 

### Contributions
1) Improved the performance of its baseline architecture, PGGAN
2) Improved Disentanglement -> Handle individual features
3) Publicated High-resolution dataset, FFHQ

### Some Concepts
#### AdaIN (Adaptive Instance Normalization)
1) Information on content and style can be extracted through VGG Encoder. 
2) 'Style' can be represented by statistics of feature space (mean and variance here)
3) So changing those statics alter the style of an input 

$$
\begin{aligned}
& AdaIN(x,y)=\sigma{(y)}(\frac{x-\mu{(x)}}{\sigma{x}}) + \mu{(y)} \\
& \\ 
& \mathit{, \ where \ x \ is \ feature \ of \ image \ for \ content, \ so \ } \\
& \\
& \mathit{to \ remove \ style \ of \ content, \ do: \} (\frac{x-\mu{(x)}}{\sigma{x}}) \\
& \\
& \mathit{to \ apply \ style \ from \ y, \ do:  \ } \sigma{(y)}(\frac{x-\mu{(x)}}{\sigma{x}}) + \mu{(y)} \\
& \\
& \mathit{make \ sure \ that \ these \ changes \ should \ be \ done \ in \ \mathbf{feature \ space} }  \\
\end{aligned}
$$

Encoder is not learnable. Only Decoder is trained, which means to learn how to invert features genereated from AdaIN to image space through Decoder

##### Formulation
- T = Style Transfer Network (Encoder-AdaIN-Decoder)
- f = encoder (the front of pre-trained VGG-19 (~relu4_1))
- g = Decdoer to learn
Given those terms, feature t generated through AdaIN layer is like following

$$t=\operatorname{AdaIN}(f(c), f(s)) $$

Decoder, g, generates image T(c,s) to which style is applied, while learning how to invert t to image space

$$T(c, s)=g(t)$$

Continue: https://lifeignite.tistory.com/48?category=460776


#### VGG (Very Deep Convolutional Networks for Large-Scale Image Recognition)
VGG19 consists of 19 weight layers (16 CNN, 3 FC).  
High performance with simple structure.  
Impact of 'deep' layer structure has been identified    
Use 3x3 filters so that expedites learning speed and increases non-linearity. 
  - Resulting size feature map is same when we do 3x3 convolution three times and when one time 7x7)
  - This means less parameters to learn at once and more non-linearity due to more activations
  - So Having more convolution enables model to fit loss function faster and to increase non-linearity

#### Loss: WGAN-GP
Traditional GAN shows instable convergence of loss. To cover this, Wasserstein GAN had been introduced.  
WGAN stablized learning process using Lipshitz constraint and weight clipping for the constraint   
WGAN-GP adopted gradient penalty to fullfill the requirement for Lipshitz
