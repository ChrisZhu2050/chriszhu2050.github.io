---
layout: page
title: "Parameter-Efficient Fine-Tuning"
# permalink: /llm-distillation/
---  
- LoRA (Low-Rank Adaptation)   
    > LoRA will train $\Delta W$, the existed attention's W will not be touched.  

    > $
    \Delta W = BA \in \mathbb{R}^{d \times k},\quad   
    B \in \mathbb{R}^{d \times r},\quad
    A \in \mathbb{R}^{r \times k}
    $  

    In the forward pass, the $\Delta W$ will participate the attention's calculation, and in the backward pass, only $\Delta W$ will be updated during the LoRA fine-tune.   
    <br>
    $\Delta W$ can be integrated to the exited W in order to avoid the additional calculation.  
    <br>

    >LoRA majorly for adjust the tone or output standard of LLM, for training domain LLM, CPT is needed for adding additional knowledge to the weights. 

    
***  

- QLoRA  
  with LoRA, frozen weights of base model will still consume GPU, for example a 7B model, the weights will occupy 14G GPU even frozen.  
  Three key technology of QLoRA help to optimize it:  
  1. NF4 (4-bit NormalFloat)
     - Compress the weights from 16-bit to 4-bit   
  2. Double quantification  
     - Compress the Scaling constant  
  3. Paged Optimizers  
     - may use CPU to balance he workload of GPU to avoid the out of memory


***  

- Adapter Tuning  
- DPO  
- Prefix/Prompt Tuning  
- RETRO
- MemLLM