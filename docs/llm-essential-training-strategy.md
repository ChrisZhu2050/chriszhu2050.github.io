---
layout: page
title: "Training Strategy"
---  

- Warmup  
    - At beginning of training, gradually increase the learning rate  
- Learning Rate Annealing  
    - At the end of training, gradually decrease the learning rate  

***  

- Cosine Annealing  
    - High learning rate at beginning, and gradually decrease during the training period and increase again at the end of training  
- WSD (Warmup with Cosine Decay)  
    - At the beginning of training, use the small learning rate and warmup it to the predefined one, after that perform the annealing  