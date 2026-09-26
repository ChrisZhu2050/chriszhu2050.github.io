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
      | Public Industry Dataset | [IndustryCorpus 2](https://huggingface.co/datasets/BAAI/IndustryCorpus2)<br>[Pile](https://pile.eleuther.ai/)<br>[Hugging Face Datasets](https://github.com/huggingface/datasets) and [Sources](https://huggingface.co/datasets)     <br>arXiv / OpendataLab / IEEE /ACM / patent database (e.g.USPTO) |
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
      ```mermaid
         graph LR
            B("`Language classification  `") --> C("`Primary Processing Transformation`") --> D("`Tagging`")  --> E("`Customized Filtering Strategy`") --> F("`Quality Scoring / Filtering `") --> H("`Text Duplication`")--> I("`Manual Quality Check Test`")
            click A "#Forward-Transformer"
      ```  
      - Language classification  
        - Cleaning of non-target languages  
      - Primary Processing Transformation  
         - Remove email addresses, URL links, phone numbers  
          - Remove Emoji emoticons  
          - Transfer Markdown to HTML and extract text  
          - Filter out extremely short texts (e.g. < 10 words) and extremely long texts without line breaks  
         - Transfer text to UTF-8 without byte order mark  
         - Remove the ASCII 0-31 and 128-159 code and zero-width characters in Unicode  
         - Standardize spaces and punctuation  
            - e.g. replace 160 to normal space(32)  
         - Remove the unicode Zero-width characters  
            - e.g. U+200B/U+200C/U+FEFF/U+200E  
      - Tagging
        -  Perplexity Filtering  
           > Calculate perplexity for the corpus via LLM (e.g. GPT-2, LLAMA), base on the calculated PPL and filter the very high / low text  
      - Customized Filtering Strategy   
        - Compliance and privacy removal (e.g. PII)  
      - Quality Scoring / Filtering  
         > The Model for scoring the corpus to identify the quality and then filtering   
         - [FastText](https://fasttext.cc/)
         - [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)  
         - LLM-as-judge (e.g. GPT-4) 
          
       
      - Text Duplication  
      
         | Level | Method | Comments | 
         |:----:|:----:|:----:|
         | File | SHA-256 | Avoid duplicate file with different name |
         | Phase | Bloom Filter + Hashset | Use Hashset double confirm the Bloom indicated duplication case |
         | File | MinHash+LSH |  Transfer file to shingles, and calculate Jaccard similarity |  

      - Manual Quality Check Test  
        - Validate with the long tail cases 
        - Manually prepare the basement dataset(Gold Standard) for evaluating the cleaning process  
        - An manual check example by Data Annotator:  
          1. Define the checklist (e.g. no HTML mark left / none TECH related subject)  
          2. Layering Sampling (e.g. from blog sourcing data random select 100 items)  
          3. Training of Data Annotator(e.g. define precisely what's none TECH related)  
          4. Perform the data checking in data annotate platform (e.g. Argilla, Label Studio, CVAT)  
          5. Analyze the verification result (e.g. Severe residual HTML in the data of blog source data)  
          6. Optimizing & Iteration (e.g. enhance the filtering for blog sourcing data)  

      - Other Tools  
         - [DataTrove (HF)](https://github.com/huggingface/datatrove) 
  


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
