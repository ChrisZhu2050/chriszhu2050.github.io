---
layout: page
title: "Continued Pre-Training"
# permalink: /llm-distillation/
---  
- Before Start  
    ```mermaid
            graph LR
                A("`Domain Corpus Token
                `") --> C{"`If >100m
                `"}--NO-->D(SFT/RAG)
                C--YES-->E(Continued Pre-Training)-->F{IF Weight >100B}--NO-->G(Unsloth / Hugging Face Transformer / Meta Chinchilla)
                F--YES-->H(Megatron-LM)
                
    ```
   
    - [Unsloth](https://unsloth.ai/)
    - [Hugging Face Transformers](#https://huggingface.co/docs/transformers/en/index) + [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)  
    - [Meta Chinchilla](metachinchilla.com)  
    - [Megatron-LM](#https://github.com/nvidia/megatron-lm)    
    - [Training Data >>](/docs/llm-essential-training-data/#training-data-cpt)   

---  
- Dataset of Validation   
    - Loss of industry validation set  
        - Extract 2% ~ 5% high quality corpus as the validaton set (must exlcude from Industry training corpus)
        - Low frequency (e.g. run on each 500 steps) since comparision is needed   

    - Loss of general validation set  
        - [SlimPajama validation split](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)  
        - [MMLU](https://crfm.stanford.edu/helm/mmlu/latest/)  

        - 1k~10k pieces of text are enough for loss calculation. High frequency monitoring will be done because only rely on the loss calculation  
<br>  
- Transformer Components  
    > Learning Rate: 1e-5 ~ 5e-5: Higher than SFT, much lower than pre-training(4e-3)  
    - Vocabulary extension  
        - tokenizer  
            BPE (Byte-Pair Encoding) for Tokenizer extension  
        - special_tokens_map   
<br>  
-  Monitor & Adjustment 
     - Baseline of estimation  
         - Loss of Domain train
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

- Other Appoaches
    - Model Merging  

*(7B-14B recommended?)*
 <a href="" id="whereami"></a> 