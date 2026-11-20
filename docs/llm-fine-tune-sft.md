---
layout: page
title: "Supervised Fine-Tuning"
# permalink: /llm-distillation/
---
>SFT is for teaching the model dialogue format and task execution ability.   
Different capability needs different training dataset 

```mermaid
        flowchart LR
         a("Target")
         b("Capability 1")
         c("Capability n")
         b1("Baseline 1")
         b2("Training Data 1")
         b21("Data Labeling 1")
         c1("Baseline n")
         c2("Training Data n")
         c21("Data Labeling n")  
         d("PEFT")  
         f("Assesment")  
         e("Deployment")

         a --> b
         b --> b1
         b --> b21
         b21 --> b2
         a --> c
         c --> c1
         c --> c21
         c21 --> c2
         b2 --> d
         c2 --> d  
         d --> f 
         b1 --> d 
         c1 --> d 
         f --> e
  ```  

***  
>Target  

<a href="" id="whereami"></a>  

***  
>Capability & Baseline 

***  
>Data Labeling  

***
>PEFT (Parameter-Efficient Fine-Tuning)  


- LoRA (Low-Rank Adaptation)   
    > LoRA will train $\Delta W$, the existed attention's weights will not be touched.  

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

- QLoRA (Quantized LoRA)  
  With LoRA, frozen weights of base model will still consume GPU, for example a 7B model, the weights will occupy 14G GPU even frozen.  
  Three key technology of QLoRA help to optimize it:  
  1. NF4 (4-bit NormalFloat)
     - Compress the weights from 16-bit to 4-bit   
  2. Double quantification  
     - Compress the Scaling constant  
  3. Paged Optimizers  
     - may use CPU to balance he workload of GPU to avoid the out of memory  


- DoRA (Weight-Decomposed LoRA)  

***  
>Assesment    
***  
>Deployment  
- vLLM  
- llama.cpp

***  
Quick experiment tool normally is not flexible on some specific needs (e.g. the loss weights adjustment on training corpus)  

- [HuggingFace PEFT](https://huggingface.co/docs/peft/index) + [Transformers](https://huggingface.co/docs/transformers/index)  
- LLaMA-Factory(WebUI)  
