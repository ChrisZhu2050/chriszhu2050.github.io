---
layout: home
title: "LLM Essential"
---

  
***  
  
GPT Training Process via Transformer ([State of GPT](https://karpathy.ai/))           [[>>]](/docs/llm-fine-tune-cpt/#whereami)

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
## Glance
- Learnable weight that a model only have one:

    | Component | Weight | Shape | 
    |:----:|:----:|:----:|
    | Token Embedding | $W_E$ | [voca, d] |
    | Final RMSNorm | $γ_{final}$ | [d]|
    | LM Head | $W_{LM}$| [voca, d] |
 

- Learnable weights that each layer contains one ( N layers )  

    | Component | Weight | Shape | 
    |:----:|:----:|:----:|
    | RMSNorm1 | $γ_1$ | [d] |
    | Attention | $W_Q$ | [d, d]|
    | Attention | $W_K$ | [d, d] |
    | Attention | $W_V$ | [d, d] |
    | Attention | $W_O$ | [d, d] |
    | RMSNorm2  | $γ_2$| [d] |
    | SwiGLU | $W_{gate}$| [d, $d_{ff}$] |
    | SwiGLU | $W_{up}$| [d, $d_{ff}$] |
    | SwiGLU | $W_{down}$| [$d_{ff}$, d] |
    
- AdamW (Optimizer) 
  > AdamW needs to store each leanable weights's first moment and second moment for optimizing the weights updating direction and length. 
  Means during the backprogation, AdamW need to maintain addtional 2 same shape weight for each leanble weights  

    \[
    \underbrace{mn}_{W{_{gradient}}}+\underbrace{mn}_{M{_{1st moment}}}+\underbrace{mn}_{V{_{2nd moment}}}=3mn
    \] 
     e.g. 7B modle's CPT need at least 58GB GPU:
  - 14GB for trainable parameters
  - 42GB for Adam weight gradients and optimizer states 
  - 2GB for activations  

