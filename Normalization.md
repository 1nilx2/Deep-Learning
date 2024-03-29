continue: https://lifeignite.tistory.com/46

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

#### BN (Batch Normalization)
So to alleviate ICS, Batch Normalizaing has been suggested to be applied before Activation

$$
\begin{aligned}
& \mathbf{Input: \ } \mathit{Values \ of \ x \ over \ a \ mini-batch: \ } B = \{x_i...m\} \\
& \mathit{\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ Parameters \ to \ be \ learned: \gamma,\beta} \\
& \mathbf{Output: \ } \{ y_i=BN_{\gamma,\beta}(x_i) \} \\
& \mu_{B} \leftarrow \frac{1}{m} \sum_{i=1}^{m}{x_i} \\
& \sigma^2_B \leftarrow \frac{1}{m}  \sum_{i=1}^{m}{(x_i-\mu_{B})^2} \\
& \hat{x_i} \leftarrow \frac{x_i-\mu_{B}}{\sqrt{\sigma^2_B + \epsilon}} \\
& y_i \leftarrow \gamma \hat{x_i} + \beta \equiv BN_{\gamma,\beta}{(x_i)} \text{\ \ \ \ \ (scale and shift)} \\
\end{aligned}
$$

##### Learnable Parameters $\mathbf{\gamma, \beta}$

![image-2](https://user-images.githubusercontent.com/88100984/224215823-7cc5d14d-a1b9-41c4-9b8d-2c5ab89001ba.png)

BN is located before activation, so it can reduce non-linearity of Activation layer(tanh). 
That's why $\gamma \ and \ \beta$ are learnable so that values can be scaled and shifted appropriately.

##### Why Model with BN performs better?
Becuase BN does not have impact on reducing ICS, other explanation is needed.   
One reason can be found by BN's characteristic of smoothing solution space.  

![image](https://user-images.githubusercontent.com/88100984/224217376-ebd9d85e-e40a-42cc-891c-a0429430b056.png)

Smoother solution space allows higher learning rate with faster and stable convergence.

Ref: https://www.notion.so/Batch-Normalization-0649da054353471397e97296d6564298
