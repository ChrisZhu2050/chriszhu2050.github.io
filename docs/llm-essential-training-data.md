---
layout: page
title: "Training Data"
---  
   - Raw data of base model's training 
     - [Common Crawl](https://commoncrawl.org/)
     - [C4/C4.EN](https://github.com/google-research/text-to-text-transfer-transformer/tree/main#c4) (filtered from April 2019 snapshot of Common Crawl )
     - Github / [Wikipedia](http://wikipedia.org) / [ArXiv](https://arxiv.org/) / [Stack Exchange](https://stackexchange.com/)
     - Books 1/2/3 (digital books)  
    <br>  
   - Training Data of SFT model  
      Manually prepared by people  

      | Question | Answer |
      |:----:|:----:|
      | Prompt 1 | Expected Response 1 |
      | Prompt 2 | Expected Response 2 |
      | Prompt 3 | Expected Response 3 |
      | ... | ... |  
       
   - Training Data of RW model  
      Manually prepared and rank by people  

        | Question | Answer | Rank |
        |:----:|:----:|:----:|
        | **Prompt 1** | Response 1 from LLM | <\|Reward\|> |
        | **Prompt 1** | Response 2 from LLM | <\|Reward\|> |
        | **Prompt 1** | Response 3 from LLM | <\|Reward\|> |
        | ... | ... | ... |  
