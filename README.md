
Outline:
- llm essential
- llm fine-tune
- agent framework
- data feedback(?)

***

## LLM essential
### Model training
```parameters(e.g. 13b) = Weight + Bias```
![](img\2026-07-28-18-21-28.png)


### Transformer
Encoder
1. data input & embedding
2. positional encoding
3. multi-head attention
4. add & normalize
5. feed forward neural network
6. add & normalize
7. multiple encoder
   
Decoder
1. outputs
2. output embedding
3. positional encoding
4. masked multi-head attention
5. add & normalize
6. multi-head attention
7. add & normalize
8. feed forward neural network
9. add & normalize
    
Linear & Softmax

Output probabilities
