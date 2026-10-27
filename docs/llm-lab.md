---
layout: page
title: "Lab"
---  

## Glance  
- Evolution sources  
  
    | Data | Handling Approach | Expectation | 
    |:----:|:----:|:----:|
    | New conversation data | Memory | Memorable |
    | New operation/live data | CPT + DPO | Optimization |
    | Interaction feedback on the output<br>(good/bad) | DPO | Optimization |
    | New high quality domain dataset | CPT+SFT+DPO | Inferable |
    | New major version of base model | CPT+SFT+DPO | Inferable |


<br>  

- Iteration Pace

    | Time Period | Approach | Data | Expectation | 
    |:----:|:----:|:----:|:----:|
    | Daily | RAG | Conversation data | Memorable |
    |Weekly | DPO | New feedback on the result | Memorable |
    | Bi-Weekly | CPT+SFT/DPO/LoRA | Accumulated feedback on the result <br>(good/bad)<br> Accumulated new domain corpus<br>Accumulated new operation / live corpus| Optimization |
    | Quarterly | CPT+SFT+DPO | Accumulated new feedback/domain/operation/live corpus with latest base model<br>(exclude expired data)  | Inferable |   

    *(Period of above depends on base model's iteration and new generated data volume)*

