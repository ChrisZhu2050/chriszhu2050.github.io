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
        - [Unsloth](https://unsloth.ai/)
        - [Hugging Face Transformers](#https://huggingface.co/docs/transformers/en/index) + [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)  

    - Weights \> 100B  
        - [Megatron-LM](#https://github.com/nvidia/megatron-lm)    


- Data 
    - Corpus
        - Industry corpus & General corpus  
            - 5:5 or 8:2 for avoiding the Catastrophic Forgetting  
        - Learning Rate: 1e-5 ~ 5e-5  
            Higher than SFT, much lower than pre-training 
    - Cleaning & Deduplication
        - Apache NiFi
        - DuckDB  
        - MinHash
        - BPE(Byte-Pair Encoding) for Tokenizer extension

- Vocabulary extension  
    - tokenizer  
    - special_tokens_map  

- Estimate & Monitor  
    - MLflow  
    - DVC  
- Deployment
    - vLLM  

*(7B-14B recommended?)*