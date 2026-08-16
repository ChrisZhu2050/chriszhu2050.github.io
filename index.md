---
layout: home
title: "LLM Essential"
---

  
***  
  
GPT Training Process via Transformer ([State of GPT](https://karpathy.ai/))

```mermaid

graph LR
    A(Training Data)  --> B(Weights Initialization) --> C(Tokenizer Pipeline) --> D(Training Forward Pass) --> E(Training Backward Pass)
    click A "#training-data"
    click B "#weights-initialization"
    click C "#tokenizer-pipeline"
    click D "#training-forward-pass"
    click E "#training-backward-pass"
```     

***  
## Training Data  
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

   

***  

## Weights Initialization


  ```mermaid
      graph LR
          A("`Token Embedding`")  --> C("`Self-Attention`") --> D("`FFN`") --> E("`LayerNorm`") --> F("`LM Head`")
  ```  

  1. Token Embedding  
      <br>
      > Only one in a model!  

      Matrix Shape:  
      > V × d<sub>model</sub>  

      (*V for Vocabulary size*)
         
      e.g.  
      > V = 32000  
      d<sub>model</sub> = 512  

      Initialize via **Truncated Normal Distribution**:    

      > mean = 0  
      std = 0.02 or 
      $\frac{1}{\sqrt d_{model}}$   
      a: -2.0 * std  
      b: 2.0 * std  
      Data Range: [-0.04, 0.04]
     
      Example:  

      > $$
      \begin{pmatrix}
      v_{1} & d_{2} & d_{3} & ... & d_{512} \\
      v_{2} & 0.0123 & -0.0045 & ... & 0.0289 &\\
      ... & ... & ... & ...& ...\\
      v_{32000} & -0.0312 & 0.0008 & ... & 0.0156 &
      \end{pmatrix}
      $$  
      

      After initialize:  
        > $  
          mean \approx 0\\    
          std  \approx 0.02  
          $  
  
      <br>  

  2.  Self-Attention (W<sub>Q</sub> / W<sub>K</sub> / W<sub>V</sub> /  W<sub>output</sub>)   
      <br>
      > Each layer has one!  

      W<sub>Q</sub> / W<sub>K</sub> / W<sub>V</sub> Matrix Shape:  

      IF head = 1:  

      > d<sub>model</sub>  × d<sub>model</sub>   

      IF head>1:  
      > d<sub>model</sub>  ×  $\frac{d_{model} }{head}$   


      W<sub>Q</sub> / W<sub>K</sub> / W<sub>V</sub>  Normal Initialization:  

      > W ~ N(0,σ<sup>2</sup>)  
          
      Above expression means: W follows a normal distribution with mean 0 and variance σ<sup>2</sup>  

        | Model | σ | Residual Scaling |
        |:----:|:----:|:----:|
        | GPT 3 | 0.02 |  ? | 
        | LLaMA 2 | 0.02 | ? | 
        | DeepSeek V3 | 0.006 | ? | 


      W<sub>Q</sub> / W<sub>K</sub> / W<sub>V</sub> example:  

      > $$
        \begin{pmatrix}
        d_{1} & d_{2} & d_{3} & ... & d_{512} \\
        d_{2} & 0.0610 & -0.0521 & ... & 0.0289 &\\
        ... & ... & ... & ...& ...\\
        d_{512} & -0.0312 & 0.0008 & ... & 0.0156 &
        \end{pmatrix}  
        $$   

      <br>  

      Residual Depth Scaling:  

      W<sub>Output</sub> Matrix Shape:     
      > d<sub>model</sub>  × d<sub>model</sub>  
      
      <br>  

      > σ<sub>residual</sub> = $\frac{σ}{\sqrt 2N}$    
      *N => number of transformer layers*  

      <br>  

      W<sub>output</sub> Initialization:  
      > W<sub>output</sub> ~ N(0,($\frac{σ}{\sqrt 2N}$)<sup>2</sup>) 

      <br>

  3. FFN (Feed Forward Network)  

      ```mermaid
      graph LR
          G("`Attention
          (d<sub>model</sub>,d<sub>model</sub>)`")  --> 
          A("`W<sub>1</sub>
          (d<sub>model</sub>,d<sub>ffn</sub>)`")  --> B("`Activation`") --> D("`W<sub>2</sub>
          (d<sub>ffn</sub>,d<sub>model</sub>)
          `") --> E("`Output
          (d<sub>model</sub>,d<sub>model</sub>)`")
      ```   
        > Each layer has one!

      - Two linear transformation (W<sub>1</sub>, W<sub>2</sub>)   
    
          > d<sub>ffn</sub> = d<sub>model</sub>*4  

          <br>

          W<sub>1</sub>:  
          Matrix shape:     
          > d<sub>model</sub>  × d<sub>ffn</sub>  

          Initialization:  
          > W<sub>1</sub> ~ N(0,$\sqrt \frac{2}{d_{model}}$)   

          <br>

          W<sub>2</sub>:  
          Matrix shape:  
          > d<sub>ffn</sub> × d<sub>model</sub>  

          Initialization:    
          > W<sub>2</sub> ~ N(0,$\sqrt \frac{2}{d_{ff}}$) 

      - an activation in between  
        - ReLU/GELU

        <br> 

  4. MoE (Mixture of Expert)  
      ```mermaid
      flowchart LR
          
          G("`token x1
          (Hidden State x1,x2,x3,...)
          `")  --> 
          A("`Router
          (g<sub>i</sug>=x<sub>i</sub>W<sup>R</sup>)
          p<sub>i​</sub>=Softmax(g<sub>i</sub>)
          [0.05,0.02,0.38,...]
          `")  --> B("`Expert 1
          *(GPU 1)*
          `") & B1("`Expert 2
          *(GPU 6)*
          `") & B2("`Expert ...
          *(GPU ...)*
          `") &  B3("`Expert n
          *(GPU n)*
          `") --> D("`Routed Experts
            Top2(g1​)={E1,E3}
          `") --> E("`Expert 1
          *(FFN -> W<sub>1.1</sub>,W<sub>2.1</sub>,W<sub>3.1</sub>)*
          SwiGLU(?)
          `") & E1("`Expert 3
          *(FFN -> W<sub>1.3</sub>,W<sub>2.3</sub>,W<sub>3.3</sub>)*
          SwiGLU(?)
          `") --> F("`Combine
          y1​=p<sub>1</sub>E1​(x1​)+p<sub>3</sub>E3​(x1​)
          （y1,y2,y3,...）
          `") 
      ```   
      Weight initialization of FNN of each experts is same as normal FNN:
      > W~N(0, σ<sup>2</sup>)  
  <br>   

  5. LM Head  
      > logits = h*W<sub>head</sub><sup>T</sup>  

      *head is hidden state and W<sub>head</sub> is LM Head's weight*

      - Independent initialize via W~N(0, σ<sup>2</sup>)(e.g. GPT-3, LLaMA)
      - Share Weight with Token Embedding for save parameter purpose?(e.g. Bert, GPT-2)  
  <br>   
 

***  


## Tokenizer Pipeline 
  
  ```mermaid
      graph LR
          A("`Normalizer`")  --> B("`Pre-tokenizer`") --> C("`Subword Algorithm`") --> D("`Vocabulary`") --> E("`Post-processor`") --> F("`Token ID Mapping`")
  ```   
  
  1. Normalizer    
      Perform Unicode normalization,  deduplication, or special symbol cleaning to ensure consistent formatting of input text

      ```mermaid
        quadrantChart
          title 4 methods of unicode normalization
          x-axis Decomposition --> Composition
          y-axis "Canonical" --> "Compatibility(Aggressive)"
              quadrant-1 "(e.g. ﬁ → fi / ① → 1)"
              quadrant-2 "(e.g. ﬁ → f+i / ① → 1)"
              quadrant-3 "(e.g. é → e + ́ )"
              quadrant-4 "(e.g. e + ́ → é)"
          "NFKD": [0.25, 0.8]
          "NFKC": [0.75, 0.8]
          "NFD": [0.25, 0.25]
          "NFC": [0.75, 0.25]
      
      ```
          NFC: Normalization Form Canonical Composition  
          NFD: Normalization Form Canonical Decomposition  
          NFKC: Normalization Form Compatibility Composition (used by LLM training)  <==   
          NFKD: Normalization Form Compatibility Decomposition  

      Normally LLM training was not performing the most aggressive normalize because it cause the meaning losing on letters  
    <br>

  2. Pre-tokenizer  
      It's for transferring the corpus to initial chunks for calculating token.  

      Regex-based pre-tokenization (GPT used), Ġ present the space:   
        ```mermaid
            graph LR
                A("`I'm learning AI!`") --> B("`I
                'm
                Ġlearning
                ĠAI
                !`") 
        ```  

      <br>
      SentencePiece's pre-tokenizer, ▁ present the space: 
      
      ```mermaid
            graph LR
                A("`Hello world`") --> B("`▁Hello ▁world`") 
      ```
      <br>

      Byte-level transferring: 
      ```mermaid
            graph LR
                A("`Chinese: 中`") --> B("`UTF-8: E4 BD A0`") 
      ```   
        <br>

  3. Subword Algorithm  

      - BPE ([Byte-pair encoding](https://github.com/tpn/pdfs/blob/master/A%20New%20Algorithm%20for%20Data%20Compression%20(1994).pdf))  
        ```mermaid
          graph LR
              A(a b c d b c d) --> B("`a **X** d **X** d`") --> C("`a **Y**`")
        ```
        > Above letters are for illustrating, they're maybe unicode bytes in reality.  
        "a bcd" are the final 2 tokens, and "bcd" will be added to the vocabulary with token id.  
      - BBPE ([Byte-level BPE](https://arxiv.org/abs/1909.03341))  
        ```mermaid
          graph LR
              A("`UTF-8 byte
              e.g. B8`") --> B("8 Bit => 
              1 0 1 1 1 0 0 0") --> C("`0/1 of each bit =>
              2<sup>8</sup> = 256 possible byte values`") --> D("`Decimal/Hex just for reading more easier
              e.g. 10111000 => 184 => B8`")
        ```
        > Initial vocabulary only have these 256 basic bytes.  
          Need to transfer the text to NTF-8 byte firstly, then implement BPE for vocabulary iteration and tokenization.  

      - WordPiece ([Paper](https://arxiv.org/abs/1810.04805))  
        ```mermaid
            graph LR
                A("`low → l, ##o, ##w
                lower → l, ##o, ##w, ##e, ##r`") --> B("Use below formula to calculate each pair") --> C("`low → l, ##o, ##w
                lower → l, ##o, ##w, **##er**`") --> D("`Vocab:
                subword(low, ##er, ##ing etc.)
                `")
          ```

        > $$\text{score}(a, b) = \frac{\text{freq}(a, b)}{\text{freq}(a) \times \text{freq}(b)}$$
        
          > Used in BERT (Encoder Only).  
          Advantage in search / RAG /Edge LLM(Size is smaller)   
      - Unigram ([Paper](https://arxiv.org/abs/1808.06226))
        ```mermaid
            graph LR
                A("`voca (>100k)=>
                letters + subwords

                `") --> B("` p(x)= $$\frac{freq(x)}{freq(x)+...+freq(n)} $$`") --> C("`  
                $$ \mathbb{E}[\text{count}(x)] $$ and recalculate the p(x) 
              
                `") --> D("`loss(x)  and  maximum $$ \mathcal{L} = \sum_{S \in \mathcal{D}} \log P(S) $$
                `")
          ```
          > It proceed from result(raw data) to reason(voca+probability) with big initial vocabulary.  
          Vocabulary pruning will be done with ME iteration and loss calculation.  

        <br>
    
  4. Vocabulary  

      - Training Data  
        Subset of Raw data of base model training(e.g. BookCorpus, Wikipedia)
    
      - BBPE(Byte-level BPE)  
        initialized with all possible 256 UFT-8 byte values and finalized with frequent bytes pair and so on

        | TokenID | Byte | Hex | Comments |
        |:----:|:----:|:----:|:----:|
        | 0 | 0 | 00 | byte 0 |
        | 1 | 1 | 01 | byte 1 |
        | ... | ... | ... | ... |
        | 255 | 255 | FF | byte FF |

      - BPE   
        Initialized with characters, numbers and so on from Raw data   

      - WordPiece  
        Initialized with basic letters and finalized with characters with/without **##**, numbers and so on from Raw data

        | TokenID | Character |
        |:----:|:----:|
        | 0 | play |
        | 1 | ##ing |
        | 2 | ##er|
        | ... | ... |
        | 30522| ##able |

      - Unigram  
          Initialized with big vocabulary(>100k) and prune to target size(e.g. ~30k)
          Specific letter to indicate space:
        **▁This ▁is ▁a ▁tokenizer**  
  
        | TokenID | Character |
        |:----:|:----:|
        | 0 | ▁This |
        | 1 | This |
        | 2 | Th|
        | ... | ... |
        | 500001| ▁tokenizer |    

      <br>

  5. Post-processor  
      
        - Add structural mark to the token sequence (such as `<bos>`, `<eos>`)
    
        - Add Chat Template mark likes `<|user|>`、`<|assistant|>`    
    
          | TokenID | Special mark |
          |:----:|:----:|
          | 111 | `<bos>` |
          | 112 | `<eos>` |
          | 113 | `<\|user\|>`|
          | 114 | `<\|assistant\|>`|
          | ... | ... |
  
          ```mermaid
            graph LR
                A("`BBPE
                [e4 bd a0,  e5 a5 bd]`") --> B("Post-processor
                [ e4 bd a0,  e5 a5 bd, < bos >]
                ") --> C("`Token ID Mapping`") 
          ```
          Post-processor may also happen after the Token ID Mapping, depends on different mechanism of models     
        <br>

  6. Token ID Mapping  

      ```mermaid
          graph LR
              A("`Token`") --> B("`Vocabulary`") --> C("`Token ID`") --> D("`Transformer`") 
      ```  
      <br>
  7. Tokenizer Toolkit  
      - [SentencePiece](https://github.com/google/sentencepiece)
      - [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) 
      - [tiktoken](https://github.com/openai/tiktoken)
    
    
      ```mermaid
          kanban
          [Tokenization Toolkit]
              [Normalizer]
              [Pre-tokenizer]
              [Subword Trainer / Algorithm]
              [Vocabulary]
              [Post Processor]
              [Token-ID mapping]
      ```  
      <br> 
  8. Popular models' tokenizer and vocabulary info  

      | Model | Company | Tokenizer | Vocabulary Size | Comments |
      |:----:|:----:|:----:|:----:|:----:|
      | GPT-4o | OpenAI | tiktoken | ~199,998 | [>> link](https://github.com/tryAGI/Tiktoken/blob/main/data/README.md?utm_source=chatgpt.com) |
      | Mixtral| Mistral AI | SentencePiece | ~32,000 | ?|
      | LLaMA 3| Meta | tiktoken-based | ~128,256 | [>> link](https://github.com/meta-llama/llama-models/blob/main/README.md?utm_source=chatgpt.com) |
      | Qwen3| Alibaba | HuggingFace Tokenizers | ~151,669 | [>> link](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md?utm_source=chatgpt.com) |
      | DeepSeek-V3| DeepSeek | SentencePiece/HF compatible implementation | ~128,000 | [>> link](https://arxiv.org/abs/2412.19437?utm_source=chatgpt.com) |   

***  

## Training Forward Pass

  ```mermaid
        graph LR
            A("`Tokenization`") --> D("`Transformer`") --> I("`Cross Entropy Loss`") 
  ```
  1. Tokenization  
  
      ```mermaid
              graph LR
                  A("`Raw Data`") --> B("`Tokenizer`")--> D("`Sequence Packing`")  
      ```  
      > *Refer => [Raw Data](#training-data) and [Tokenizer](#tokenizer-pipeline)*  

        Sequence Packing:  
          Training Sequence Length = 4096  

      ```mermaid
              graph TD
                  A("`Doc A
                  (2000 Tokens)
                  `") --> C("`Sequence 4096
                  [A1 A2 A3... *< EOS >*] + [B1 B2 B3... *< EOS >*] + [C1, C2, C3...] = 4096
                  `")
                  B("`Doc B
                  (1000 Tokens)
                  `")--> C
                  D("`Doc C
                  (2000 Tokens)
                  `")--> C
      ```
     Batch Size:  
      > GPU's capacity limit the batch size!

      Example:  
      > Training Sequence Length T = 4096  
      Micro Batch Size B = 8  

      Then **one GPU per one forward** will process => 8 × 4096 =32768 *Tokens*  
        
      IF we want:
      > Effective Batch Size=32  

      Means after 32 batches then update the weight.  
      So we need to set: 
      > Gradient Accumulation Steps = 4  

      and finally:  

      > B<sub>global</sub> = B<sub>micro</sub> × N<sub>GPU</sub> × GAS  
      Tokens<sub>step</sub> = B<sub>global</sub> × T  

      <br>  



  2. GPT's transformer  
     - Decoder  
       1.  output embedding
       2.  positional encoding
       3.  LayerNorm
       4.  masked multi-head attention
       5.  LayerNorm
       6.  feed forward neural network
       7.  Linear & Softmax  
       8.  Output probabilities  
     
     - Encoder (*not used by GPT*)  
  <br>


  3. cross-entropy Loss  
    - logits --> softmax/sigmoid --> cross-entropy

      > $$ L(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$  

  - Gradient computation  
    > $$ \nabla L = [\frac{\partial L}{\partial w_1} + \frac{\partial L}{\partial w_2} +..... \frac{\partial L}{\partial w_n}] $$  


***  
## Training Backward Pass  
  


  - Weight update by Adam
      > $$ W_n = W_o - η*\nabla L(W_o) $$       
   



<!-- [![](images\1.png)](https://chriszhu2050.github.io/images/1.png) -->
<!-- ![](https://chriszhu2050.github.io/images/1.png) -->
<!-- <br> -->

<!-- ## LLM Fine-tune  
***
## LLM Distillation  
***
## Agent Framework  
***

## Data Flywheel
*** -->

