---
layout: page
title: "Parameter-Efficient Fine-Tuning"
# permalink: /llm-distillation/
---  
- LoRA (Low-Rank Adaptation)   
 > LoRA is training on $\Delta W$, the existed attention's W will be frozen.  

  > $
    \Delta W = BA \in \mathbb{R}^{d \times k},\quad   
    B \in \mathbb{R}^{d \times r},\quad
    A \in \mathbb{R}^{r \times k}
    $  

In the forward pass, the $\Delta W$ will participate the attention's calculation, and in the backward pass, only $\Delta W$ will be updated during the LoRA fine-tune.  

>LoRA majorly for adjust the tone or output standard of LLM, for training domain LLM, CPT is needed for adding additional knowledge to the weights. 

    
***  

- QLoRA
- Adapter Tuning
- DPO
- Prefix/Prompt Tuning  
- RETRO
- MemLLM