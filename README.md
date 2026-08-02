
Outline:
- LLM essential
  - [Model training](#section1)
  - [Transformer](#section2)
- [LLM fine-tune](#section3)
- [Agent framework](#section4)
- [Data feedback](#section5)

***

## LLM essential  
- Model training  
<a id="section1"></a>
[![](images\1.png)](https://chriszhu2050.github.io/images/1.png)
<!-- ![](https://chriszhu2050.github.io/images/1.png) -->

  - Forward pass
    - Data compression (Byte-pair encoding)
    
  - Backward pass  
    - Loss computation  
        > cross-entropy loss $$ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$


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
  
## LLM fine-tune  
<a id="section3"></a>  

## Agent framework  
<a id="section4"></a>  

## Data feedback  
<a id="section5"></a>  
