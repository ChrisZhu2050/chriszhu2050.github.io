---
layout: home
title: "LLM Essential"
---

  
***  
  
GPT Training Process via Transformer ([State of GPT](https://karpathy.ai/))           [[>>>]](/docs/llm-essential-backward-pass/#whereami)

```mermaid

graph LR
    A(Training Data) --> B(Weights Initialization) --> C(Tokenizer Pipeline) --> D(Training Forward Pass) --> E(Training Backward Pass)
    click A "/docs/llm-essential-training-data/"
    click B "/docs/llm-essential-weights-initialization/"
    click C "/docs/llm-essential-tokenizer-pipeline/"
    click D "/docs/llm-essential-forward-pass/"
    click E "/docs/llm-essential-backward-pass/"
```     

***  
### Glance
- Only one in a model:

    | Component | Weight | Shape | 
    |:----:|:----:|:----:|
    | Token Embedding | $W_E$ | [voca, d] |
    | Final RMSNorm | $γ_{final}$ | [d]|
    | LM Head | $W_{LM}$| [voca, d] |
 

- One per layer(N layers)  
    | Component | Weight | Shape | 
    |:----:|:----:|:----:|
    | RMSNorm1 | $γ_1$ | [d] |
    | Attention | $W_Q$ | [d, d]|
    | Attention | $W_K$ | [d, d] |
    | Attention | $W_V$ | [d, d] |
    | Attention | $W_O$ | [d, d] |
    | RMSNorm2  | $γ_2$| [d] |
    | SwiGLU | $W_{gate}$| [d, $d_{dff}$] |
    | SwiGLU | $W_{up}$| [d, $d_{dff}$] |
    | SwiGLU | $W_{down}$| [$d_{dff}$, d] |

