---
layout: page
title: "Lab"
---  

## Glance  
- Evolution sources  
    | Data | Handling Approach | Expectation | 
    |:----:|:----:|:----:|
    | New conversation data | Memory | Memorable |
    | Feedback of the result<br>(good/bad) | DPO | Optimization |
    | New high quality domain dataset | CPT+SFT+DOP | Inferable |
    | New major version of base model | CPT+SFT+DOP | Inferable |


<br>  

- Iteration Pace

| Time Period | Approach | Data | Expectation | 
|:----:|:----:|:----:|:----:|
| Weekly | SFT/DPO | Accumulated feedback on the result <br>(good/bad) | Searchable |
| Monthly | LoRA | Accumulated new domain data | Memorable |
| Quarterly | CPT+SFT+DPO | Accumulated data with latest base model<br>(exclude expired data)  | Inferable |   

*(Period of above depends on base model's iteration and new generated data volumn)*

