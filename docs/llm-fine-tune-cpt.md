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
        - [Meta Chinchilla](metachinchilla.com)  

    - Weights \> 100B  
        - [Megatron-LM](#https://github.com/nvidia/megatron-lm)    


- [Training Data >>>](/docs/llm-essential-training-data/#training-data-cpt)


- Transformer Components
    - Vocabulary extension  
        - tokenizer  
        - special_tokens_map  

- Catastrophic Forgetting  
- Estimate & Monitor  
    - Baseline of estimation
    - MLflow  
    - DVC  
- Deployment
    - vLLM  
- Other Appoaches
    - Model Merging  

*(7B-14B recommended?)*
 <a href="" id="whereami"></a> 