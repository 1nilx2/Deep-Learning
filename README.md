# Deep-Learnings

## Revisit Learning / Training of Deep learning
How to update parameters(𝜃, weights and biases)?


$\theta^* = argminL(f_\theta(x),y)$

$\mathit{Iterative \ Method \ } \theta^* = argminL(f_\theta(x),y) = argminL(\theta)$

|   Questions  |  Strategies  |
|--------------|--------------|
|How to update $\theta \rightarrow \Delta\theta$ | Only if $L(\theta+\Delta\theta) < L(\theta)$  |
|When we stop to search?  | If $L(\theta + \Delta\theta) == L(\theta)$  |
|How to find $\Delta\theta$ so that $L(\theta + \Delta\theta) < L(\theta)?$|$\Delta\theta = -\eta\nabla L$, where $\eta>0$|

------------------------------------------------------------------------------------------------------------------------------

**Taylor Exapnsion** $\rightarrow \ L(\theta + \Delta\theta) = L(\theta) + \nabla L \cdot \Delta\theta$ + second derivative + third derivative + ... (1)

**Approximation** $\ \ \ \rightarrow L(\theta + \Delta\theta) \approx L(\theta) + \nabla L \cdot \Delta\theta$ (2)

$L(\theta + \Delta\theta)-L(\theta) = \Delta L =  \nabla L \cdot \Delta\theta$ (3)

If $\Delta\theta = -\eta \nabla L,$ then $\Delta L = -\eta||\nabla L||^2 < 0$, where $\eta > 0$ (4)

------------------------------------------------------------------------------------------------------------------------------

That's why we 
  - take **Negative** direction of gradient  --- (4)
  - apply learning rate of small value (due to approximation not use full expansion) --- (2)

## Back Propagation
<img width="578" alt="스크린샷 2023-04-04 오후 9 57 47" src="https://user-images.githubusercontent.com/88100984/229969762-6ff469b4-a7ee-49c3-849c-5a5277a90085.png">


## Revisit Loss Function
### View-Point 1: Backpropagation
Assume that we use sigmoid as activation function
  - Cross Entropy is expected to perform better 
  - $\delta_{CE} = \nabla_a C \odot \sigma'(Z^L) = \frac{a-y}{(1-a)a}(1-a)a = a-y$




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
