---
layout: page
title: "Continued Pre-Training"
# permalink: /llm-distillation/
---  
> CPT is for supplimenting model's specific domain knowledge (e.g. legal / medical science)  


- Transformer Components  
    > Learning Rate: 1e-5 ~ 5e-5: Higher than SFT, much lower than pre-training(4e-3)  
    - Vocabulary extension (*Optional*)  
        - tokenizer  
            BPE (Byte-Pair Encoding) for Tokenizer extension  
        - special_tokens_map   
<br>  
-  Monitor & Adjustment 
     - Baseline of estimation  
         - Loss of domain validation set
         - Loss of General validation set
     - Dynamic proportion of corpus  
         Adjust proportion of domain & general corpus base on loss monitoring, here's the example:  
         1. At beginning use proportion 9:1 (Industry corpus : Wikipedia corpus) for quickly learn the industry knowledge  
         2. Adjust to 7:3 later for avoiding the Catastrophic Forgetting  
     - MLflow  
     - DVC  
> Corpus 20 times than volume of Weights of base model may get best convergent effect (?)  
e.g. 8B base model need 160B tokens industry data   

<br>  

- Deployment   
    - vLLM  

---  

- Low GPU solution
  - Hugging face transformer + DeepSpeed  
    - reduce batch size  
    - enable gradient acccumulation  
    - enable activation checking  
    - enable ZeRO-3 CPU offload    
  - [GaLore](https://github.com/jiaweizzhao/galore)  
    - a gradient projection method, compress gradient matrix and reduce the GPU consumption
  
 <a href="" id="whereami"></a> 