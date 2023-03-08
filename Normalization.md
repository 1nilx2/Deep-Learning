\[ 
\text{$x=4$ or $x=5$} 
\]


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

$\mathbf{Input: \ } \mathit{Values \ of \ x \ over \ a \ mini-batch: \ } B = \{x_i...m\}$

$\mathit{\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ Parameters \ to \ be \ learned: \gamma,\beta}$

continue
https://www.notion.so/Batch-Normalization-0649da054353471397e97296d6564298