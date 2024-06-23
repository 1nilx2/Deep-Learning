# NLP

## RNN
I want to make a prediction. If I can reflect previous information, could the result be better?  
How to reflect the previous information?

$$h_t = f(W_{xh}x_t + W_{hh}h_{t-1})$$
$$y_t = g(W_{hy}h_t)$$
$$f(\cdot) = tanh, g(\cdot) = softmax$$

## LSTM
 - has been suggested to alleviate the problem of long-term dependency at RNN
 - Why that happened? 
    - Due to simply conveying previous information using hidden state ($h_t$)
    - $h_t = W_{hh}h_{t-1} + W_{xh}x_t$
 - How to handle
    - Sophisticate operations around $h_t$ so that model can discern what to remember and to forget

### Key concept
Cell state and three gates of forget, input, and output.  
Don't be confused by the word, gate. There are just vectors

 - Cell state: processed input
 - Forget gate: how to forget previous information, $c_{t-1}$
 - Input gate: how to use current temporary information, $\tilde{c}$
 - Output gate: how to process and return the current information ($c_t$) in conjuction with forget and input gates.

$$<Calculate \ Gates>$$
 $$f_t = \sigma(W_{xh_f}x_t + W_{hh_f}h_{t-1} + b_{h_f})$$
 $$i_t = \sigma(W_{xh_i}x_t + W_{hh_i}h_{t-1} + b_{h_i})$$
 $$o_t = \sigma(W_{xh_o}x_t + W_{hh_o}h_{t-1} + b_{h_o})$$

$$<Update \ Cell \ state>$$
 $$\tilde{c_t} = tanh(W_{xh_g}x_t + W_{hh_g}h_{t-1} + b_{h_g})$$

$$<Update \ Hidden \ state>$$
 $$h_t = o_t \otimes tanh(c_t)$$


### Conclusion
Each LSTM layer generates hidden state and cell state, applicable to short-term  and long-term information respectively. By doing so, it tries to handle to the long-term depency, which was hardly dealt in RNN Strcture.

However, LSTM also still suffers from the vanishing/exploding gradient, causing ineffective and inefficient learning. 

Look at Transformer to know how the architecture tries to solve the problem referring to entire sequence and adopting skip connection.