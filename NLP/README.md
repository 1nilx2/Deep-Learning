# LSTM
## Background
RNN is about $h_t \ and \ x_t \rightarrow$ Problem of Long-term dependency.

    - It does not keep information of tokens (or vector of those)

## Main
LSTM is about $\mathit{cell \ state}$ and $\mathit{3 \ gates}$  
    - $\mathit{forget} \sim C_{t-1} \sim (h_{t-1}, x_t)$  
    - $\mathit{input} \sim h_{t-1}, x_t$  
    - $\mathit{output} \sim C_t, h_{t-1}, x_t$  

How much forget, input, and ouput is 'ratio', so take 'sigmoid'

|   Gates  |  Others  |
|--------------|--------------|
|$i_t = \sigma (W_{hi}h_{t-1} + W_{xi} x_{t-1} + b_i)$ | $g_t=\mathit{tanh} (W_{hg}h_{t-1} + W_{xg} x_t + b_g)$ |
|$f_t = \sigma (W_{hf}h_{t-1} + W_{xf} x_{t-1} + b_f)$ | $C_t = f_t \odot C_{t-1} + i_t \odot x_t$ |
|$o_t = \sigma (W_{ho}h_{t-1} + W_{xo} x_{t-1} + b_o)$ | $h_t = o_t \odot \ tanh(C_t)$ |

### Dimensions

$x_t = (d_{embedding}, \ )$

$W_{x \textunderscore } = (d_{hidden \textunderscore state}, d_{embedding})$

$W_{h \textunderscore } = (d_{hidden \textunderscore state}, d_{hidden \ state})$

$C_t \ or \ h_t = (d_{hidden \textunderscore state}, \ )$


# Transformer

## Background
Recurrent model computes 'sequentially'. This shown difficulty in parallelization, decreased performance with long sequence, and limitation of batch due to memory usage (Some remedies exist like factorization tricks and conditional compution).

Transformer and, thus, attention mechanism has been suggested to overcome these problems

## Diagram
<img width="1889" alt="스크린샷 2023-04-16 오후 7 06 13" src="https://user-images.githubusercontent.com/88100984/232351308-257d3233-ea0a-4d7e-ace3-e2cb8081fa24.png">

<img width="766" alt="Layer_Normalization" src="https://user-images.githubusercontent.com/88100984/232352006-68e3e095-ce3c-4dc5-abd8-e71d4036f6b1.png">


### Seq2Seq
![seq2seq](https://user-images.githubusercontent.com/88100984/227757491-94779534-447e-40bf-aa64-c332c675d398.jpg)

### Note
![Transformer_1](https://user-images.githubusercontent.com/88100984/227757504-f1743c32-049b-4377-becf-a6c062bfa41c.JPG)

![Transformer-2](https://user-images.githubusercontent.com/88100984/228073192-8fde7f83-3f0c-4b59-80c9-3e9c7421b3ad.jpg)


## Multi-head attention (n x scaled dot-product attention) [Encoder]

Input vector = input embedding + positional encoding/embedding
* encoding is given like sinusoid functions
* embedding is learnable 

Input vector -> input vector * 3(Q,K,V) -> linear -> split by num_head -> 
  -> Multi-head attention -> Concat -> Linear -> Residual Connection & Layer Normalization

$(n, d_{model}) \rightarrow (3, n, d_{model}) -> W^Q, W^K, W^V $

$\mathit{In \ Encoder, \ the \ input \ vector \ used \ for \ generating \ Q, K, V}$

$\mathit{Input \ Vector \ (input \ embedding \ + \ positional \ embedding/encoding), } (n, d_{model})$

$----Linear_1 \ \mathit{input \ vector} \cdot W^Q $

$----Linear_2 \ \mathit{input \ vector} \cdot W^K$

$----Linear_3 \ \mathit{input \ vector} \cdot W^V$

$\rightarrow Q, K, V \ \ \ {n, d_{model}/num_head}$




## FFNN
- Dense -> Activation -> Dense 
- $(n, d_{model}) \cdot (d_{model}, d_{ff}) \rightarrow \mathit{activation} \rightarrow (n, d_{ff}) \cdot (d_{ff}, d_{model})$
- residual connection: $h(x) = x + f(x)$ 
- Layer Normalization: $LN(h(x)) = LN(x + f(x)) \leftarrow \hat{x_{i}} = \frac{x_i - \mu_x}{\sigma_x}$  

ref: https://happy-jihye.github.io/nlp/

# BERT
BERT (Bidirectional Encoder Representation of Transformer)
  - Takes encoder of Transformer 
  - 'Bidirectional' due to the fact that BERT does not use look-ahead mask while decoder of Transformer does 
  - What decoder of Transformer does is left to users who will fine-tune it by adding post-layers
  - 'Pre-trained Large Language Model (LLM)' which can be fine-tuned according to users' ends 

## Input representation
  - Token Embedding: Sub-word tokenization (WordPiece)  [input_ids]
  - Segment Embedding: 0 and 1 but all tokens can be zero if it's for sentence prediction [token_type_id]
  - Position Embedding
  - Attention Mask
    - 1 for meaningful tokens (CLS, SEP, Other normal tokens) 
    - 0 for others (tokens for padding)

## Training
  - MLM: Masked Language Model -> Masking some tokens  
  - NSP: Next Sentence Prediction -> Infer whether two sentences are connected 

## Fine-tuning
Single Sentence Classification -> CLS Token  
Tagging -> Every tokens between CLS and SEP  
Text Pair / Regression -> CLS Token  
Question-Answering -> Tokens between first and second SEP tokens  

* We can assume that [CLS] token represent overall information of the sentence 
* With this idea, Sentence BERT can be fine-tuned


# LLaMA
Open and Efficient Foundation Language Models
  - Considers 'Inference' cost not just training cost
  - 13B model is competitive with 175B GPT-3
  - Trained with public data <> Chinchilla, PaLM, GPT-3
  - Tokenizer: BPE (Byte-pair encoding) via SetntencePiece ==> 1.4T Tokens

## Few-shot properties
An ability to perform new task 
  - from textual instructions or few examples
  - which appears when scaling up models to sufficient size

This motivated approaches that is based upon increasing model size (parameters). 
However, recent work shows that best performance can be achieded by smaller models trained on more data. 

If this is effective, we can think that  
**'we haven't utilized full-potential each parameter has'**


## Architecture
Transformer-based but has some difference
   - Pre-normalization
   - SwiGLU activiation instead of ReLU
   - Removed absolute positional embedding. Instead, add rotary positional embedding

## Main Results
Two types of guidance were given to models
  - Zero-shot: a textual description only
  - Few-shot: few examples 

