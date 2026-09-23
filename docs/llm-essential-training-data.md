---
layout: page
title: "Training Data"
---  
   - Data of base model's pre-training 
     - [Common Crawl](https://commoncrawl.org/)
     - [C4/C4.EN](https://github.com/google-research/text-to-text-transfer-transformer/tree/main#c4) (filtered from April 2019 snapshot of Common Crawl )
     - Github / [Wikipedia](http://wikipedia.org) / [ArXiv](https://arxiv.org/) / [Stack Exchange](https://stackexchange.com/)
     - Books 1/2/3 (digital books)  
    
   <a href="" id="training-data-cpt"></a>
   - Data of continued pre-training   
    
      - Domain data   

      |  | Industry Corpus |
      |:----:|:----|
      | Public Industry Dataset | [IndustryCorpus 2](https://huggingface.co/datasets/BAAI/IndustryCorpus2)<br>[Pile](https://pile.eleuther.ai/)<br>Hugging Face Datasets / arXiv / OpendataLab / IEEE /ACM / patent database (e.g.USPTO) |
      | Publication| Professional textbooks and guides(e.g. CPA, PMP)<br>Industry Analytic Report <br>Official published data from government <br>Industry Books |
      | Vertical community | (e.g. StackOverflow/GitHub) |
      | Synthetic Data | Synthetic big corpus via small corpus by [EntiGraph algorithm](https://arxiv.org/abs/2409.07431) |
      | Enterprise Privacy Data | Operational Data (e.g. ERP / CRM / MES )<br>Internal docs( Manual / SOP / Product intro)<br>Customer interaction data( chat / ticket / text of phone call )<br> R&D data ( design / code / bug / testing )  | 

      - General data  
         - [SlimPajama Dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B) -- [Introduction](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)  
         - [Wikipedia Dataset](https://huggingface.co/datasets/wikimedia/wikipedia)
         -  Dolmino Mix 1124 - Dedicated for Mid-training  
         - [Common Crawl](https://commoncrawl.org/) 

   ---  

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
         - Filter out the repetition rate of a certain word or phrase in a piece of text is excessively high (e.g. 30%)
      - Quality Classifier  
         Train the caterize small model(e.g. FastText)
      - Perplexity Filtering  
         use small N-gram to calculate the PPL and filter the very high / low text  
      - Compliance and privacy removal (e.g. PII) 
      - MinHash (Deduplication)
      - Tools  
         - [DataTrove (HF)](https://github.com/huggingface/datatrove) 
         - NeMo Curator (NVIDIA)
         - Open Source Pipeline (RedPajama/FineWeb) 

   ---

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
