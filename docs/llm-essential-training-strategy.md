---
layout: page
title: "Training Monitoring"
---  


- Indicators of training result    
  
    | Phase | Data | Indicator | Expectation | 
    |:----:|:----:|:----:|:----:|
    | Entire | Training Dataset | Training Loss | Monotonically decreasing |
    | Entire | Validation Dataset | Validation Loss | Monotonically decreasing |
    | Entire | Validation Dataset |Perplexity | Perplexity = exp(loss)<br>Low PPL doesn't mean better, need to monitor the trend with validation dataset and validate in parallel on if catastrophic forgetting is happening |
    | Entire | Training Dataset | Plateau | The loss curve enters a flat range, doesn't mean should stop the training, recent research shows during this period silent learning is happening behind  |
    | Entire | Training Dataset | Gradient Norm | Good experience shows low gradient and high loss at beginning is the positive signal of better final performance model, may base on this to stop the poor quality training |


- Dataset of domain validation   
    - Extract 2% ~ 5% high quality corpus as the validaton set (must exclude from domain training corpus)
    - Low frequency (e.g. run on each 500 steps) since comparision is needed   

- Dataset of general validation  
    - [MMLU](https://crfm.stanford.edu/helm/mmlu/latest/)  
    - [SlimPajama validation split](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) 
    - 1k~10k pieces of text are enough for loss calculation. High frequency monitoring will be done because only rely on the loss calculation  

***  
- Warmup  
    - At beginning of training, gradually increase the learning rate  
- Learning Rate Annealing  
    - At the end of training, gradually decrease the learning rate  
- Early Stopping  
    - According to the above indicators and stop the training when unexpected result shows(e.g. patience=5 and validation result still negative after that)  

***  

- Cosine Annealing  
    - High learning rate at beginning, and gradually decrease during the training period and increase again at the end of training  
- WSD (Warmup with Cosine Decay)  
    - At the beginning of training, use the small learning rate and warmup it to the predefined one, after that perform the annealing  