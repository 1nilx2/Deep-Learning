# LLM

## FP16
Intuitive vibe it gives is something is havled from FP32.
In a numbering system with FP32, 32 bits are needed to express one number, 1bit used for sign, 8bit for exponent, and 23 bit for mantissa.

| Precision \ Item  | Sign | Exponent |  Mantissa  |
|-------|--------|---------|----------|
| FP32  | 1 bit  |   8bit  |   23 bit   |
| FP16  | 1 bit  |   5bit  |   10 bit   |


How it's actually different from FP32? 
  - Numerical Range a precision level can express is different, which might result in difference especially in large/accumulated matrix multiplication due to round-off 
    - FP16: $ -6.55 \times 10^{4}  \text{ to }  6.55 \times 10^{4} $
    - FP32: $-3.4 \times 10^{38} \text{ to } 3.4 \times 10^{38}$
  - Memory usage / Speed accordingly will be less required at a cost of precision. 
  - However, this does not mean that FP16 does always impact a model's performance significantly. If our use case requires less sophisticated computation, the repercussion small delta brings can be minimized.
  - Or we might experience a better training due to the increased batch size. 

### Mixed-precision
Is there a way to cherry-pick both FP16 and 32? We might also consider mixed-precision. The assence is to use FP16 for most of the forward/backward operations, while FP32 is applied to operations requiring more accurate computation such as updating weights and gradient accumulation.

In addition, the loss calculated by FP16 will be scaled up. I should remebmer that the range FP16 can express is narrower, which might lead to a vanishing gradient.

![Mixed-precision](figs/mixed-precision.png)


## Lora
Low-rank adpatation which tries to approximate, with rather simple decomposition, what would've been done by full fine tuning, $\Delta W$. 

LoRA is a kind of adapter so we can easily exchange it according to our usage. If we have a strong foundation model and different downstream tasks on top of the model, we can tune model for individual tasks then change adapters for purpose.

![LoRA](figs/LoRA.png)

You can approximate $\Delta W \text{ through } BA$
Also the impact of adapter can be handled by specifying $\text{decomposition level}, r, and \ \alpha$
  
We can also speicfy to which matrix/layer LoRA will be applied. 

What is appropriate level of adaptation and which layer could be positively influenced by it?
 - It seems that choosing variety of layers has more impact increasing the rank. 
 - Always, this depends on our specific task and the gap between foundation model and the domain in interest. So practical investigation is definitely recommneded.

![LoRA2](figs/LoRA2.png)
