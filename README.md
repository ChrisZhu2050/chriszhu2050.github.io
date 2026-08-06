Outline:
- [LLM essential](#llm-essential)
- [LLM fine-tune](#llm-fine-tune)
- [Agent framework](#agent-framework)
- [Data feedback](#data-feedback)

***

## LLM essential
<!-- <a id="section1"></a> -->
1. Model training process
[![](images\1.png)](https://chriszhu2050.github.io/images/1.png)
<!-- ![](https://chriszhu2050.github.io/images/1.png) -->
  

2. Training Data
     - Raw data for base model
       - Sources
         - Common Crawl
         - C4/C4.EN (from April 2019 snapshot of Common Crawl)
         - Github/Wikipedia/ArXiv/Stack Exchange
         
     - prompt&answer for SFT model
       - repaired manually by people
     - promt&result with rank for RW model
       - prepaired manually by peope     
  <br>
3. Weights initialization
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
    
4. Forward pass
    - Tokenization
      - Algorithm
        - BPE (Byte-pair encoding)
      - Vocabulary
        - Base vocabulary(e.g. 256 UTF8 code)
        - Increment via BAE(Iteration to add each most frequent subword to the vocabulary)
        - Typical vocabulary size: LLaMA(~32K-100K ?) / Deepseek(~65K-250K?)
      - Tokenizer
        - SentencePiece(tool)
    

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

1. Backward pass  
    - cross-entropy Loss computation  
        - logits --> softmax/sigmoid --> cross-entropy

        > $$ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

    - Gradient computation
        > $$ \nabla L = [\frac{\partial L}{\partial w_1} + \frac{\partial L}{\partial w_2} +..... \frac{\partial L}{\partial w_n}] $$

    - Weight update by Adam
        > $$ W_n = W_o - η*\nabla L(W_o) $$  
        

  <br>

## LLM fine-tune  
***

## Agent framework  
***

## Data feedback  
***
