---
layout: page
title: "Continued Pre-Training"
# permalink: /llm-distillation/
---  
> CPT is for supplimenting model's specific domain knowledge (e.g. legal / medical science)  


- [GaLore (Gradient Low-Rank Projection)](https://github.com/jiaweizzhao/galore)  
    > GaLore is a low GPU CPT solution， it project the gradient to low rank in order to reduce the GPU consumption of AdamW  

    SVD (Singular Value Decomposition) means any matrix 
    $
    A=UΣV^T
    $  
    - Shape of A is (mxn) 
    - Shape of U is (mxm) 
    - Shape of Σ is (mxn) 
    - Shape of $V^T$ is (nxn) 

    > The elments on the main diagonal of Σ are singular value.  

    Example:  

    $
    \Sigma = \begin{bmatrix}
    \sigma_1 & 0 \\
    0 & \sigma_2 \\
    0 & 0 \\
    0 & 0
    \end{bmatrix}
    $  

    location (1, 1) = $\sigma_1$  
    location (2, 2) = $\sigma_2$  
    others = 0

    So the singular value is [$\sigma_1$, $\sigma_2$]  

    > Keep the top K largest singular values and their corresponding vectors to get the best low-rank approximation.  

    > When sort singular value in descending order of Σ, meanwhile U and V will also swith in order to align with sort of Σ. 

    > If target is keep top k(e.e. keep 95% of singular value), then need to talor Σ, U and V.  

    - Shape of $Σ_k$ is (k x k) 
    - Shape of $U_k$ is (m x k) 
    - Shape of $V_k^T$ is (k x n) 
    - Shape of $A_k = U_kΣ_kV_k^T$  is (m x n)  
    > A and $A_k$ has same shape (m x n), but we didn't store $A_k$, insteadly we store $U_k$,$V_k^T$ and $s_k$.  

    Here're their shapes:  
        $U_k$: (m x k)  
        $V_k^T$: (k x n)  
        $s_k$: k numbers array (e.g. [$\sigma_1$, $\sigma_2$...$\sigma_n$])   

    > Follow-up calculation tasks will wrok on the k dimention, and will reduce the computational load.  

***    

- DAPT (Domain-Adaptive Pre-training)  


***   

  - Hugging face transformer + DeepSpeed  
    - reduce batch size  
    - enable gradient acccumulation  
    - enable activation checking  
    - enable ZeRO-3 CPU offload    

***  

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
  
 <a href="" id="whereami"></a> 