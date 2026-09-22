---
layout: page
title: "Training Data"
---  
   - Raw data of base model's pre-training 
     - [Common Crawl](https://commoncrawl.org/)
     - [C4/C4.EN](https://github.com/google-research/text-to-text-transfer-transformer/tree/main#c4) (filtered from April 2019 snapshot of Common Crawl )
     - Github / [Wikipedia](http://wikipedia.org) / [ArXiv](https://arxiv.org/) / [Stack Exchange](https://stackexchange.com/)
     - Books 1/2/3 (digital books)  
   <br>  
   <a href="" id="training-data-cpt"></a>
   - Data of continued pre-training  
      - Inudustry data  
         - [IndustryCorpus 2](https://huggingface.co/datasets/BAAI/IndustryCorpus2) 
         - [Pile](https://pile.eleuther.ai/)  
         > Synthetic big corpus via small corpus: [EntiGraph algorithm](https://arxiv.org/abs/2409.07431)
      - General data
         - [SlimPajama Dataset](https://modelscope.cn/datasets/swift/SlimPajama-627B/files) -- [Introduction](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)  
         - [Common Crawl](https://commoncrawl.org/)  
         - Dolmino Mix 1124 - Dedicated for Mid-training  

         > Industry corpus & General corpus：  
         5:5 or 8:2 for avoiding the Catastrophic Forgetting  

         > Learning Rate: 1e-5 ~ 5e-5:   
         Higher than SFT, much lower than pre-training  

         *(20 times of Weights of base model may get best convergent effect e.g. 8B base model need 160B tokens industry data ?)*  
      <br>
   - Data Cleaning & Deduplication  
      - Encoding & character standardization
         - Transfer text to UTF-8 without byte order mark  
         - Remove the ASCII 0-31 and 128-159 code and zero-width characters in Unicode  
         - Standardize spaces and punctuation  
            - e.g. replace 160 to normal space(32)  
         - Remove the unicode Zero-width characters  
            - e.g. U+200B/U+200C/U+FEFF/U+200E
      - Strip HTML/XML tags using regular expressions  
         - Excluding URL links, email addresses, phone numbers and Markdown special syntax characters  
         - Remove Emoji emoticons
      - Cleaning of non-target languages  
      - Length and basic statistical filtering  
         - Filter out extremely short texts (e.g. < 10 words) and extremely long texts without line breaks  
         - Filter out the repetition rate of a certain word or phrase in a piece of text is excessively high (e.g.30%)



         - MinHash(Deduplication)
         - BPE(Byte-Pair Encoding) for Tokenizer extension  
         - Apache NiFi

 
   <br>  

   - Training Data of SFT (Supervised Finetuning) model  
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
