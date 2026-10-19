---
layout: page
title: "LLM Fine-tune"
# permalink: /llm-fine-tune/
---
 
 <a href="" id="whereami"></a> 
## Glance
  - CPT (Continued Pre-Training) 
  - SFT (Supervised Fine-Tuning)  
  - DPO (Direct Preference Optimization)
  - RLHF (Reinforcement Learning from Human Feedback)
  - PERT (Parameter-Efficient Fine-Tuning)
    - LoRA
    - QLoRA
    - Adapter Tuning
    - Prefix/Prompt Tuning  
  - Full Fine-Tuning  

  ***  
- How to choose 
    ```mermaid
            graph LR
                A("`Domain Corpus Token
                `") --> C{"`If >10M
                `"}--NO-->
                D1{If Syntheticed Data will work?}--YES-->D2("`Synthetic Data
                (e.g. EntiGraph algorithm)`")
                D1--NO-->D3{"` IF > 1000+ Q&A records `"}--YES-->D4("`SFT`")--100+ better/worse group-->D6("`DPO`")
                D3--NO-->D5("`PERT / RAG`")
                C--YES-->E(Continued Pre-Training)-->F{"`If base on large-scale model?
                ( e.g. >100B )`"}--NO-->G( Hugging Face Trainer )
                F--YES-->H(NVIDIA NeMo Framework / Megatron-LM
                )
                
    ```  
    - [Hugging Face ( Transformers + Accelerate )](https://github.com/huggingface) + [DeepSpeed](https://github.com/deepspeedai/DeepSpeed )  

        - Accelerate of Hugging Face is for covering multi-GPUs/TPU/fp16 part when using PyTorch models to write the training loop  
        - DeepSpeed's ZeRO-3 + CPU Offload may help single GPU to run the major LLM model(?)
    - [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/index.html) + [Megatron-LM](#https://github.com/nvidia/megatron-lm)    
    - [Synthetic Data](/docs/llm-essential-training-data/#training-data-cpt)   

    