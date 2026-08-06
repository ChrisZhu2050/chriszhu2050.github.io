Outline:
- [LLM Essential](#llm-essential)
- [LLM Fine-tune](#llm-fine-tune)
- [LLM Distillation](#llm-distillation)
- [Agent Framework](#agent-framework)
- [Data Flywheel](#data-flywheel)

***

## LLM Essential
<!-- <a id="section1"></a> -->
1. GPT Training Process
[![](images\1.png)](https://chriszhu2050.github.io/images/1.png)
<!-- ![](https://chriszhu2050.github.io/images/1.png) -->
  

2. Training Data
     - Raw data for base model's training 
       - Sources
         - [Common Crawl](https://commoncrawl.org/)
         - C4/C4.EN (from April 2019 snapshot of Common Crawl)
         - Github/Wikipedia/ArXiv/Stack Exchange
         - Books(1,2,3)
         
     - Prompt & answer for SFT model
       - Prepaired manually by people
     - Prompt & result with rank for RW model
       - Prepaired manually by peope     
  <br>
3. Weights initializing before base model's training
    - FFN
      - Xavier/Glorot
      - Kaiming/He
    - Q/K/V/O
      - Xavier/Glorot
      - Kaiming/He
    - Bias
      - normally initial as 0
    - LayerNorm
      - γ as 1 and β as 0
    - Token & Position embedding
      - N(0,1)
    - LM Head
      - Xavier/Glorot/Kaiming
      - or shared with embedding   
   <br> 
4. Tokenizer initializing before base model's training
    - Algorithm
      - BPE (Byte-pair encoding)
      - Byte-level BPE
    - Vocabulary
      - Base vocabulary(e.g. 256 UTF8 code)
      - Increment via BAE(Iteration to add each most frequent subword to the vocabulary)
      - Typical vocabulary size: LLaMA(~32K-100K ?) / Deepseek(~65K-250K?)
    - Tool
      - SentencePiece   
 <br> 
5. Training Forward Pass
    - Tokenization
    - Transformer  
    <!-- <a id="section2"></a>   -->
        - Encoder  
          1. data input & embedding
          2. positional encoding
          3. multi-head attention
          4. add & normalize
          5. feed forward neural network
          6. add & normalize
          7. multiple encoder
        - Decoder  
          1. outputs
          2. output embedding
          3.  positional encoding
          4.  masked multi-head attention
          5.  add & normalize
          6.  multi-head attention
          7.  add & normalize
          8.  feed forward neural network
          9.  add & normalize

    - Linear & Softmax
    - Output probabilities  
  <br>

6. Training Backward Pass  
    - cross-entropy Loss computation  
        - logits --> softmax/sigmoid --> cross-entropy

        > $$ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

    - Gradient computation
        > $$ \nabla L = [\frac{\partial L}{\partial w_1} + \frac{\partial L}{\partial w_2} +..... \frac{\partial L}{\partial w_n}] $$

    - Weight update by Adam
        > $$ W_n = W_o - η*\nabla L(W_o) $$  
        

  <br>

## LLM Fine-tune  
***
## LLM Distillation  
***
## Agent Framework  
***

## Data Flywheel
***
