# Transformer

## Background
Recurrent model computes 'sequentially'. This shown difficulty in parallelization, decreased performance with long sequence, and limitation of batch due to memory usage (Some remedies exist like factorization tricks and conditional compution).

Transformer and, thus, attention mechanism has been suggested to overcome these problems

## Seq2Seq
![seq2seq](https://user-images.githubusercontent.com/88100984/227757491-94779534-447e-40bf-aa64-c332c675d398.jpg)

## Note
![Transformer_1](https://user-images.githubusercontent.com/88100984/227757504-f1743c32-049b-4377-becf-a6c062bfa41c.JPG)

![Transformer-2](https://user-images.githubusercontent.com/88100984/228073192-8fde7f83-3f0c-4b59-80c9-3e9c7421b3ad.jpg)


### Multi-head attention (n x scaled dot-product attention) [Encoder]

Input vector = input embedding + positional encoding/embedding
* encoding is given like sinusoid functions
* embedding is learnable 

Input vector -> input vector * 3(Q,K,V) -> linear -> split by num_head -> 
  -> Multi-head attention -> Concat -> Linear -> Residual Connection & Layer Normalization

$(n, d_{model}) \rightarrow (3, n, d_{model}) -> W^Q, W^K, W^V $

$\mathit{In \ Encoder, \ the \ input \ vector \ used \ for \ generating \ Q, K, V}$

$\mathit{Input \ Vector \ (input \ embedding \ + \ positional \ embedding/encoding), } (n, d_{model})$

$Linear_1 \ \mathit{input \ vector} \cdot W^Q $

$Linear_2 \ \mathit{input \ vector} \cdot W^K$

$Linear_3 \ \mathit{input \ vector} \cdot W^V$

$\rightarrow Q, K, V \ \ {n, d_{model}/num_head}$




### FFNN
- Dense -> Activation -> Dense 
- $(n, d_{model}) \cdot (d_{model}, d_{ff}) \rightarrow \mathit{activation} \rightarrow (n, d_{ff}) \cdot (d_{ff}, d_{model})$
- residual connection: $h(x) = x + f(x)$ 
- Layer Normalization: $LN(h(x)) = LN(x + f(x)) \leftarrow \hat{x_{i}} = \frac{x_i - \mu_x}{\sigma_x}$  

ref: https://happy-jihye.github.io/nlp/
