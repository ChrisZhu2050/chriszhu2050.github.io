---
layout: page
title: "Continued Pre-Training"
# permalink: /llm-distillation/
---  
- Before Start  
    - corpus tokens < 100m   
        consider SFT / RAG  
    - corpus tokens > 100m   
        Continued Pre-Training  
    - Weights \<100B  
        Hugging Face Transformers + [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)  
    - Weights \> 100B  
        Megatron-LM    


- Data 
    - Corpus
        - Industry corpus & General corpus  
            - 5:5 or 8:2 for avoiding the Catastrophic Forgetting  
        - Learning Rate: 1e-5 ~ 5e-5  
            Higher than SFT, much lower than pre-training 
    - Cleanning & Deduplication
        - Apache NiFi
        - DuckDB  
        - MinHash
        - BPE(Byte-Pair Encoding) for Tokenizer extention
 
- Vocabulary extention  
    - tokenizer  
    - special_tokens_map  

- Training frameworK  
    - Weights\<100B  
        Hugging Face Transformers + [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)  
    - Weights\> 100B  
        Megatron-LM  

- Estimate & Monitor  
    - MLflow  
    - DVC  
- Deployment
    - vLLM  
