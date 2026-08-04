
Outline:
- [LLM essential]
  - [Model training](#section1)
  - [Transformer](#section2)
- [LLM fine-tune](#llm-fine-tune)
- [Agent framework](#agent-framework)
- [Data feedback](#data-feedback)

***

## LLM essential
<a id="section1"></a>
- Model training  
[![](images\1.png)](https://chriszhu2050.github.io/images/1.png)
<!-- ![](https://chriszhu2050.github.io/images/1.png) -->
  - Training Data
    - not tagged data for base model
    - prompt&answer for SFT model
    - promt&result with rank for RW model


  - Weights initialization
      - FFN,Q/K/V/O
        - Xavier/Glorot/Kaiming/He
      - Bias
        - normally initial as 0
      - LayerNorm
        - γ as 1 and β as 0
      - Token & Position embedding
        - N(0,1)
      - LM Head
        - Xavier/Glorot/Kaiming
        - or shared with embedding



  
    
  - Forward pass
    
    - Data compression (Byte-pair encoding)
    - Transformer  
    <a id="section2"></a>  
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


  - Backward pass  
    - cross-entropy Loss computation  
        - logits --> softmax/sigmoid --> cross-entropy

        > $$ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

    - Gradient computation
        > $$ \nabla L = [\frac{\partial L}{\partial w_1} + \frac{\partial L}{\partial w_2} +..... \frac{\partial L}{\partial w_n}] $$

    - Weight update by Adam
        > $$ W_n = W_o - η*\nabla L(W_o) $$



   

        

   



## LLM fine-tune  
***



## Agent framework  
***



## Data feedback  
***


