  ---
layout: page
title: "Training Forward Pass"
---
  
  ```mermaid
        graph LR
            A("`Tokenization`") --> D("`Transformer`") --> I("`Cross Entropy Loss`") 
            click D "#Forward-Transformer"
            click I "#Forward-Cross-Entropy-Loss"
  ```
  1. **Tokenization**  
  
      ```mermaid
              graph LR
                  A("`Raw Data`") --> B("`Tokenizer`")--> D("`Sequence Packing`")  
      ```  
      > *Refer to => [Raw Data](#training-data) and [Tokenizer](#tokenizer-pipeline)*  

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

      Then one GPU per one forward will process => 8 × 4096 =32768 *Tokens*  
        
      IF we want:
      > Effective Batch Size=32  

      Means after 32 batches then update the weight.  
      So we need to set: 
      > Gradient Accumulation Steps = 4  

      and finally:  

      > B<sub>global</sub> = B<sub>micro</sub> × N<sub>GPU</sub> × GAS  
      Tokens<sub>step</sub> = B<sub>global</sub> × T   

      <br>  
      <a href="" id="Forward-Transformer"></a>
  2. **GPT's Decoder-only Transformer**  
   <br>
        Differences with original Transformer:  
        > a. No encoder  
        b. No 2nd Multiple-head attention(Cross Attention)  
        c. Post-LN instead of Pre-LN 

      ```mermaid
          block
          columns 6

            a("Token embedding")
            

          block:group2:1
            columns 1
            c1("RMSNorm") space
            b1("Q/K/V") space
            b("Positional embedding") space
            c2("Masked multi-head attention") space
            c3("Add")
          end
          block:group3:1
            columns 1
            d("2nd RMSNorm") space
            e("Feed-forward neural network") space
            h("2nd Add")
          end
          block:group4:1
            columns 1
            f("Final RMSNorm") space
            g("Linear / LM Head")

          end
          
            z("Logits")
            y("Input Of Next Layer
            or Softmax for final output")
          

          c1 --> b1
          b1 --> b
          b --> c2
          c2 --> c3
          d --> e
          e --> h
          f --> g

          a --> group2
          group2 --> group3
          group3 --> group4
          group4 --> z
          z --> y

      ```  
      <br>  

     - Token embedding  
        > About Weight, refer to [Weight initialization](#weights-initialization)  

          ```mermaid
              graph LR
                  A("`Input
                  (*Token IDs*)
                  `") --> B("`Search token in Weight and return the row vector
                  (*Weight: V x d<sub>model</sub>*)
                  `")--> D("`Output
                  *[seq_len, d<sub>model</sub>]*
                  `")  
          ```  
      <br>
      <a href="" id="layernorm"></a>  

     - LayerNorm 
          ```mermaid
              graph LR
                  A("`Input x<sub>i</sub>($$X \in \mathbb R^{B,T,d}$$)`")
                   --> 
                  B("`Calculate average(μ) & σ<sup>2</sup>`")
                  --> 
                  C("`Normalization $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^{2} + \epsilon}}$$`")
                  --> 
                  D("`Output:$$y_i = \gamma_i \cdot \hat{x}_i + \beta_i$$`")  
          ```  
          
          >Normally ϵ = 10 <sup>-5</sup>  
          Initially $γ_i=1$ and $β_i=0$ and $γ,β \in \mathbb R^{d}$, will be optimized during training  

          e.g.   
         > original x=[3,4,5]  
         output y≈[−1.2247,0,1.2247]  

         Final Output:
         >  $$
            Y \in \mathbb R^{(B,T, d)}
            $$  
         <br>  
      <a href="" id="Forward-RMSNorm"></a>  
      - RMSNorm  
        ```mermaid
              graph LR
                  A("`Input x<sub>i</sub>($$X \in \mathbb R^{B,T,d}$$)`")
                  --> 
                  C("`Normalization $$\hat{x}_i = \frac{x_i}{RMS(x)+ \epsilon}$$`")
                  --> 
                  D("`Output:$$y_i = \gamma_i \cdot \hat{x}_i $$`")  
          ```  
          > $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2}$  

          RMSNorm removed the $x_i - \mu$, because centralization is not impactting the result, only scale part will do.
      - Q/K/V Calculation
        ```mermaid
              graph LR
                  A("`Input                   x ∈ R<sup>(B, T, d)</sup>
                  `") --> 
                  B("`Q = xW<sub>Q</sub> 
                  K = xW<sub>K</sub>
                  V = xW<sub>V</sub>
                  (W<sub>Q,K,V</sub> ∈ R<sup>D×D</sup>)
                  
                  `") 
          ```  
            

        What happened in Q = xW<sub>Q</sub> ?
        >$$
        q_i = x_i \cdot W_Q = \left[ \sum_{k=1}^{D} x_{ik} \cdot W_{Q,k1},\ \sum_{k=1}^{D} x_{ik} \cdot W_{Q,k2},\ \dots,\ \sum_{k=1}^{D} x_{ik} \cdot W_{Q,kD} \right]
        $$  

        > Q, K, V ∈ R<sup>(B, T, d)</sup>  
        x<sub>ik</sub> => The k-th value of the i-th row vector of X  
        W<sub>Q,kj</sub> => the parameter in the k-th row and j-th column of the weight matrix  

        e.g.
        > D=4, x=[1,2,0,−1], W<sub>Q</sub> = 4*4 matrix  
        q<sub>i1</sub> = 1 * W<sub>Q,11</sub> + 2 * W<sub>Q,21</sub> + 0 * W<sub>Q,31</sub> + (-1) * W<sub>Q,41</sub>  
        >$$q_i \in \mathbb R^{D}$$
        
        > Q= 
        >$$
          \begin{pmatrix}
          q_{1}\\
          q_{2}\\
          ... \\
          q_{N}
          \end{pmatrix}  
          $$   

        <br>
      - Positional embedding
          > Original transformer's Sinusoidal Positional Encoding is almost deprecated:  
          X = TokenEmbedding + PositionEmbedding
          - RoPE(Rotary Position Embedding)  
            Where RoPE happens:
            > X = TokenEmbedding  
            Q = XW<sub>Q​</sub>   
            K = XW<sub>K</sub>  

            <br>  

            Then:  ​  
              > Q<sub>rot</sub>= RoPE(Q)  
              K<sub>rot</sub>= RoPE(K)  


            What's RoPE？
            > $$
                \begin{bmatrix}
                x' \\
                y'
                \end{bmatrix}
                =
                \begin{bmatrix}
                \cos\theta & -\sin\theta \\
                \sin\theta & \cos\theta
                \end{bmatrix}
                \begin{bmatrix}
                x \\
                y
                \end{bmatrix}
              $$  
            > x′= xcosθ − ysinθ  
            y′= xsinθ + ycosθ  

            Where's θ from?   
            (*all above θ is below θ<sub>p,i</sub>, don't confuse with the below RoPE Base*) 
            > θ<sub>p,i</sub> ​= p*ω<sub>i</sub>​  
            $$
            \omega_i=\frac{1}{\theta^{2i/d}}
            $$  
            ω<sub>i</sub> => Rotary frequency  
            d => Attention head dimension  
            i => A certain two-dimensional dimension of Q/K  => 0,1,2,…,d/2−1  
            θ => RoPE Base => normally the value is 10000

            Finally:

            >$$
            \mathrm{Attention}
            =
            \mathrm{Softmax}
            \left(
            \frac{Q_{rot}K_{rot}^T}{\sqrt{d_{\mathrm{head}}}}
            \right)V
            $$

      <br>  

     - Masked multi-head attention  
        
        How to reshap to *h* heads?  

        > $$
        d_k = \frac {d_{model}}{h}
        $$  

        > [B, T, d<sub>model</sub>]==>[B, T, h, d<sub>k</sub>]  
        *The calculation happened on [N, d<sub>k</sub>] dimention!*   

        Why need to do transpose?   
        e.g. t=3, h=2, d<sub>k</sub>=3
        here's the sequence of tensor in memory:  
        > $$
        \begin{pmatrix}
        p0_{T} & h{0} & C{0} \\
        p0_{T} & h{0} & C{1} \\
        p0_{T} & h{0} & C{2} \\
        p0_{T} & h{1} & C{0} \\
        p0_{T} & h{1} & C{1} \\
        p0_{T} & h{1} & C{2} \\
        p1_{T} & h{0} & C{0} \\
        p1_{T} & h{0} & C{1} \\
        ... & ... & ... &\\
        p2_{T} & h{1} & C{2} & 
        \end{pmatrix}  
        $$  

        > *C==>Component of the tensor*   

        When we caculate on h0, need gather the data of h0 from different blocks.

        > Transpose: [B, T, h, d<sub>k</sub>] => [B, h, T, d<sub>k</sub>]    

        > $$
        \begin{pmatrix}
        h{0} & p0_{T} & C{0} \\
        h{0} & p0_{T} & C{1} \\
        h{0} & p0_{T} & C{2} \\
        h{0} & p1_{T} & C{0} \\
        h{0} & p1_{T} & C{1} \\
        h{1} & p0_{T} & C{0} \\
        h{1} & p0_{T} & C{1} \\
        h{1} & p0_{T} & C{2} \\
        ... & ... & ... &\\
        p2_{T} & h{1} & C{2} & 
        \end{pmatrix}  
        $$  

        After transpose, continues data for each heads and more efficient for calculation.  

        Break-down of attention calculation:   
        > $$
        Score = Q_{rot}K_{rot}^T = [seq\_len, seq\_len]
        $$
        What's dot-product?
        >$$
        Score_{ij} = \sum_{k=1}^{d_k} (Q_{ik} \cdot K_{jk})
        $$  

        Scaling:  
        >$$
        \frac{Q_{rot}K_{rot}^T}{\sqrt{d_{\mathrm{head}}}}
        $$  
        
        Causal Mask:  
        > Shape:[T × T]  

        e.g. T = 4  
        
        >$$mask\_bool =  
        \begin{pmatrix}
        F & T & T & T\\
        F & F & T & T\\
        F & F & F & T\\
        F & F & F & F\\
        \end{pmatrix}  
        $$

        >$$M(causal\_scores) = 
        \begin{pmatrix}
        0 & −∞ & −∞ & −∞\\
        0 & 0 & −∞ & −∞\\
        0 & 0 & 0 & −∞\\
        0 & 0 & 0 & 0\\
        \end{pmatrix}  
        $$  

        > S<sub>masked</sub> = Score + causal_scores 

        >$$
        S_{masked}=
        \left(
        \frac{Q_{rot}K_{rot}^T}{\sqrt{d_{\mathrm{head}}}}+M
        \right)
        $$
        
        What's Softmax?  
        > For x=[x1​,x2​,…,xn​]
        $$
        \operatorname{Softmax}(x_i)
        =
        \frac{e^{x_i}}
        {\sum_{j=1}^{n}e^{x_j}}
        $$  

        What Softmax will do on S<sub>masked</sub>?
        > Since: $e^{−∞} →0$   
        $$
        A = Softmax(S_{masked}) 
        $$  
        
        e.g.   
        Tokens: ***I   believe  change   is   happening***     
        >$$  
        S_{masked}=
        \begin{bmatrix}
        2.0 & -\infty & -\infty & -\infty & -\infty \\
        1.0 & 2.0 & -\infty & -\infty & -\infty \\
        2.0 & 1.0 & 3.0 & -\infty & -\infty \\
        1.0 & 2.0 & 3.0 & 4.0 & -\infty \\
        2.0 & 1.0 & 3.0 & 4.0 & 5.0
        \end{bmatrix}
        $$  
      
        > $$ 
        A=Softmax(S_{masked}) = 
        \begin{bmatrix}
        1.000 & 0     & 0     & 0     & 0 \\
        0.269 & 0.731 & 0     & 0     & 0 \\
        0.245 & 0.090 & 0.665 & 0     & 0 \\
        0.032 & 0.087 & 0.237 & 0.644 & 0 \\
        0.032 & 0.012 & 0.032 & 0.087 & 0.837
        \end{bmatrix}
        $$  

        for example: ***A<sub>change</sub> ​= [0.245,0.090,0.665,0,0]***
       

        Why the above attention weights A need to multiply V?
          > Attention = A x V  
          > $$
          V=
          \begin{bmatrix}
          V_I \\
          V_{believe} \\
          V_{change} \\
          V_{is} \\
          V_{happening}
          \end{bmatrix}
          $$  
        > ***O<sub>change</sub> ​= 0.245V<sub>I</sub> ​+ 0.090V<sub>believe</sub> ​+ 0.665V<sub>change</sub>*** 

        >One of heads output like this:  
        $$
        O_i=
        \begin{bmatrix}
        O_I \\
        O_{believe} \\
        O_{change} \\
        O_{is} \\
        O_{happening}
        \end{bmatrix}
        $$

        Attention weights determine how much the V of each token contributes to the current output.  
        MHA Final Output:  

        ```mermaid
            graph TD
                Z("Input from pre-LN") -->A
                Z-->B
                Z-->D
                A("`Head1
                Output<sub>1</sub>
                `") --> C("`out_attn=Concat(Output<sub>1</sub>, O<sub>2</sub>...,O<sub>h</sub>)*W<sub>O</sub>
                `")
                B("`Head2
                O<sub>2</sub>
                `")--> C
                D("`Head3
                O<sub>3</sub>
                `")--> C
        ```
        Concat example:    
        >$
        d_{head}=4
        $  
        >$
        h=2
        $  
        >$
        O_1 \in R^{5\times4}
        $  
        >$
        O_2 \in R^{5\times4}
        $  
        >$
        Concat(O_1​,O_2​)\in R^{5\times8}
        $    

        >$
        attn = \operatorname{Concat}(O_1,\ldots,O_h)
        \in
        \mathbb{R}^{T\times d_{\mathrm{model}}}
        $

        Why need W<sub>O</sub>?  
        >concat combine all the heads outputs.    
        W<sub>O</sub> responsible for reunion the info via projection to W<sub>O</sub> weights.

     - Add  
        > $ 
        X_{out} = X_{in} + out\_attn
        $  
        > $ 
        X_{in} \in \mathbb R^{(L \times d)} 
        $  
        > $
         out\_attn \in \mathbb R^{(L \times d)} 
         $  


     - 2nd RMSNorm 
        >The input is from above Add's X<sub>out</sub>  
        >The Process may [[Refer to RMSNorm]](#Forward-RMSNorm)
     - Feed-forward neural network  
        <a href="" id="swiglu"></a>
        Most popular sturcture at this moment: SwiGLU  
        > $SwiGLU\_FFN(x) = (\text{SiLU}(xW_{gate}) \odot (xW_{up}))W_{down}$  

      

        ```mermaid
        flowchart LR
            
            A("`RMSNorm Output 
            (x ∈ R<sup>Lxd</sup>)
            `")  --> B("`Gate Path
            (W<sub>gate</sub> ∈ R<sup>d<sub>ff</sub>xd</sup>)
            `") & B1("`Value Path
            (W<sub>up</sub> ∈ R<sup>d<sub>ff</sub>xd</sup>)
            `")
            B1 --$$up=x\cdot W_{up}$$--> E
            B --$$a=x\cdot W_{gate}$$--> D("`SiLU
            g=SiLU(a)=a⋅σ(a)
            σ = Sigmoid()
            `") --> 
            E("`$$act^{L\times d_{ff}} = g \odot up$$
            `") --> F("`$$out^{L\times d_{model}} = act\cdot W_{down}$$
            `") 
        ```   
        > act => activation  
        $ d_{ff} = 8/3\times d_{model} $  

        >Shape of W<sub>gate</sub>,W<sub>up</sub> : [ d<sub>ff</sub>, d]  
        Shape of W<sub>down</sub>: [d, d<sub>ff</sub>]  
        W<sub>gate</sub>,W<sub>up</sub>,W<sub>down</sub>  ~ N(0,σ<sup>2</sup>)  
        

        > $Sigmoid(x) = \frac {1}{1+e^{-x}} $  
        *For converting the input to (0, 1)*  

        > ⊙=> Hadamard product => $c_{ij} = a_{ij} \times b_{ij} $

 

     - 2nd Add
        > $$ 
        X_{out} = X_{in} + O_{SwiGLU} 
        $$  
        > $$ 
        X_{in}, X_{out},O_{SwiGLU} \in \mathbb R^{(B,T, d)} 
        $$
     - Final RMSNorm  
        Same mechanism with previous RMSNorm, only the position is different.  
        Output is $h_{final} \in \mathbb R^{(B, T, d)}$  

     - LM Head and Logits 
        > $logits = h_{final} \cdot W_{LM}^T + b$  

        > $h_{final} \in \mathbb R^{(B,T,d)}$  

        > $W_{LM} \in \mathbb R^{(V \times d)}$  

        > $Z = logits \in \mathbb R^{(B, T, V )}$  

        e.g.   
        >$
        Z_5=
        \begin{bmatrix}
        1.2 & 0.5 & 4.8 & 0.7 & -0.3 & 1.1 & 0.2 & -1.0
        \end{bmatrix}
        $


        The logits will be the input of Next Layer input, if it's the final layer, then implement the LM Softmax:  

        $$
        P_{t,j}
        =
        \operatorname{Softmax}(Z_{t,j})
        =
        \frac{e^{Z_{t,j}}}
        {\sum_{k=1}^{V}e^{Z_{t,k}}}
        $$

        $$
        p\in \mathbb R ^{T \times V}
        $$  
        
        e.g.  

        $$
        P=
        \begin{bmatrix}
        P_{1,1} & P_{1,2} & P_{1,3} & \cdots & P_{1,V}\\
        P_{2,1} & P_{2,2} & P_{2,3} & \cdots & P_{2,V}\\
        P_{3,1} & P_{3,2} & P_{3,3} & \cdots & P_{3,V}\\
        ... & ... & ... & \cdots & ...\\
        P_{T,1} & P_{T,2} & P_{T,3} & \cdots & P_{T,V}
        \end{bmatrix}
        $$  
  

      > *Encoder is not used by GPT*   

      <br>

      <a href="" id="Forward-Cross-Entropy-Loss"></a>
  3. **Cross-entropy Loss**  
      For example if the real 5th word is "happening" and if the output of P5(happening) is:  
      > P5(happening) = 0.9 => means the loss is relatively small  
      *or*  
      > P5(happening) = 0.01 => means the loss is relatively big  

      Loss calculation:  
      > $$
        L=-\sum_{i=1}^{V}
        y_i\log(p_i)
      $$  
      > V：Vocabulary Size
      > y<sub>i</sub> is real target's one-shot
      > p<sub>i</sub> is probility of i from essimation of current model

      Since y<sub>i</sub> is a one-shot like y<sub>i</sub>​=[0,0,1,0,0,…,0], so the formular is simplified as:  
      > $
          L_i=-y_i\log p_i
        $  
      
      An example:  
      > P(happening)=0.01  
      $L=−log(0.01)\approx 4.605$   

      About loss reduction, normally use Mean, here's an example:     
      > If Batch size=4,Token=5, then  
      $$
        L_{\mathrm{batch 1}}
        =
        \frac{1}{4\times5}
        \sum_{b=1}^{4}
        \sum_{t=1}^{5}
        L_{b,t}
      $$  

      $L_{batch1}$ is input of a Backforwad, 

