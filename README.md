# Statistics

## Cross-entropy & KL Divergence
Entropy is average amount of information and can be said degree of surprise

$$
\displaylines{\mathit{for \ continuous \ x,} \\ 
\sum_{x}{p(x)log_{b}{p(x)}} = E_p[-log_{b}{p(x)}]}
$$


$$
\displaylines{\mathit{for \ continuous \ x,} \\ 
H[x]= \lim_{\Delta \rightarrow 0}{\sum_{i}{p(x_i)\Delta\ln{p(x_i)} }} = - \int{p(x)}dx}
$$

KL Divergence is the difference between two probability distributions  
More clealy, It's the difference between entropy and thus the loss due to a miss-modeling

$$
\displaylines{D_{KL}{(p||q)} = - \int{p(x)\ln{g(x)}}dx - (-\int{p(x)\ln{p(x)}dx}) \\  
=-\int{p(x)\ln{\{ \frac{q(x)}{p(x)} \}}}dx}
$$

$$
\Leftrightarrow KL(p||q) = H(p,q) - H(p)
$$



### Minimize Cross Entropy?
