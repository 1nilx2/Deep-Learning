# Deep-Learnings

## Revisit Learning / Training of Deep learning
How to update parameters(𝜃, weights and biases)?


$\theta^* = argminL(f_\theta(x),y)$

$\mathit{Iterative \ \ \ Method \ \ } \; \; \theta^* = argminL(f_\theta(x),y) = argminL(\theta)$

|Questions|Strategies|
------------------------------
|How to update $\theta -> \delta\theta$|Only if $L(\theta+\delta\theta) < L(\theta)$|



참고할만한 링크들: https://happy-jihye.github.io/nlp/

## Normalizations
### Batch Normalization
##### Role
  - re-cetering and re-scaling distribution of input feature of layers
##### Pros
  - Faster Tranining (can set learning rate higher)
  - Stable Training (less vanishing and exploding gradients)
  - Regularziation (less overfitting, more generalized)
##### Cons
  - Need to use larger size for mini-batch. Mini-batch needs to have silmiar distribution with population
##### Reason of working well
  - Alleviate ICN (Internal Covariate Shift)?
    - controversy on this. 
  - Smooth solution space of objective function
    - so far, this is strong opinion
 
#### Background
##### Normalization
  - Min-Max and Standardization (with mean and std)

##### Standardization

$$\mathit{Standardization: \ } \ x' = \frac{x-\mu{(x)}}{\sigma{(x)}} $$

Subtracting by µ(𝑥) -> zero-centered data
Dividing by 𝜎(𝑥) -> 1 ~ -1

##### Whitening
Known to be better than Standardization.
This involves  
  - reprocess features using PCA -> decorrelated features
  - conduct standardization so that all features have same scale  
But for large number of features, PCA becomes expensive

##### Covariate Shift
Due to the different distribution of training and test samples,  
the resulting functions can vary, which is called Covariate Shift 

Interncal Covariate Shift(ICS) indicates when those kinds of situation happens between inputs of layers, and
thus a model doesn't learn well. 

continue
https://www.notion.so/Batch-Normalization-0649da054353471397e97296d6564298

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

## Cross-entropy & KL Divergence
#### Entropy
Entropy is average amount of information and can be said degree of surprise

$$
\displaylines{\mathit{for \ continuous \ x,} \\ 
\sum_{x}{p(x)log_{b}{p(x)}} = E_p[-log_{b}{p(x)}]}
$$


$$
\displaylines{\mathit{for \ continuous \ x,} \\ 
H[x]= \lim_{\Delta \rightarrow 0}{\sum_{i}{p(x_i)\Delta\ln{p(x_i)} }} = - \int{p(x)}dx}
$$

#### KL Divergence
KL Divergence is the difference between two probability distributions  
More clealy, It's the difference between entropy and thus the loss due to a miss-modeling

$$
\displaylines{D_{KL}{(p||q)} = - \int{p(x)\ln{g(x)}}dx - (-\int{p(x)\ln{p(x)}dx}) \\  
=-\int{p(x)\ln{\{ \frac{q(x)}{p(x)} \}}}dx}
$$

$$\Leftrightarrow KL(p||q) = H(p,q) - H(p)$$



#### Minimize Cross Entropy? H(p,q)
When we train the model, we do differentiate KL Divergence w.r.t q  
There is no parameter for p, so just minimizing Cross Entropy will ensure same result for KL Divergence
