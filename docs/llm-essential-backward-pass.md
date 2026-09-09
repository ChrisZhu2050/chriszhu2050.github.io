---
layout: page
title: "Training Backward Pass"
---

> When got the Loss of the batch, then calculate all the Gradient with learnable Parameters ( [Initialized Weights](#weights-initialization) ) 

  ```mermaid
        flowchart LR

          a("Loss ->LM Head")
          c1("Final RMSNorm")
          b1("Residual")
          d("FFN + RMSNorm")
          d1("1st Gradient 
          Accumulation")
          e("Attention + RMSNorm")  
          e1("Residual") 
          f("2nd Gradient
           Accumulation")
          g("Layer N-1")
        
        a --> c1
        c1 --> d
        c1 --> b1
        b1 -- ∂L/∂H<sub>N</sub>--> d1
        d --∂L/∂H<sub>mid</sub>--> d1
        d1 --> e1
        d1 --> e
        e1 --> f
        e --> f
        f --> g


        

  ```  
1. **Loss -> LM Head**   
    Refer to Forward Pass, there should be 2 derivation steps :  
    > 1<sup>st</sup>:  $ \frac {\partial L}{\partial P_j} $ ( Derivative of loss calcualtion fuction )  
    2<sup>nd</sup>: $ \frac {\partial P_j}{\partial Z_i} $ ( Derivative of softmax funcion. too complex for calculation, Jacobian Matrix ??)  

    After simplified via joint derivation:  
      > $$
        \frac{\partial L}{\partial Z_i}
        =
        P_i-y_i
        $$
      > y is a one-shot, then the y<sub>i</sub>=1 here  

      Shape of $\frac{\partial L}{\partial Z}$ is same as Z=> ($T \times V$)

    $\frac{\partial L}{\partial Z_i}$ is the gradient of Softmax + Cross-Entropy, it's a scalar(e.g. -0.09).
    Nagetive means increase the logits(i) will reduce the loss, meanwhile should decrease other logits, vice versa.  

    Move on the Backward Pass as below:
    > $$
      L
      \rightarrow
      \frac{\partial L}{\partial Z}
      \rightarrow
      \left\{
      \begin{aligned}
      \frac{\partial L}{\partial H_{norm}}
      &=
      \frac{\partial L}{\partial Z}
      W_{\mathrm{LM}}^T
      \\
      \frac{\partial L}{\partial W_{\mathrm{LM}}}
      &=
      H\cdot
      (\frac{\partial L}{\partial Z})^T
      \end{aligned}
      \right.
    $$   

    <br>  

     - $\frac{\partial L}{\partial W_{\mathrm{LM}}}$ is for updating the weight of LM Head, the shape is  $[V \times d]$   

       > $
         \frac{\partial L}{\partial W_{LM}} = (\frac{\partial L}{\partial Z})^T \cdot H
         $  

         Shape of H is => [B, T, d]  

         Shape of $\frac{\partial L}{\partial Z}$ is => ($T \times V$)  

       Once above gradient is ready, next will perform the weight of LM Head udpating  

       > $W_n=W_o - η*\nabla L(W_o)$  
       *This is traditional way which is not used any more*  
       
         Most popular way:  
       - Global L2 Norm Clipping (max_norm = 1.0)  
         no independent clipping for gradiant of LM Head's weight during pre-training, because it will impact the relevance with weights of transformer.  
         An example:
           > Gradient of LM Head: [3,4] and  layer gradient: [1,2,2]  
           Global L2 Norm = $\sqrt {3^2+4^2+1^2+2^2+2^2}=\sqrt{34} \approx 5.83 $  
           Since 5.83 over the max_norm = 1.0,    
           then every gradient elements need to $\times \frac {1.0}{5.83}\approx 0,172$ to do Global Clipping   

           *(For SFT/RHDL, may have different approach)*  

        <a href="" id="Backward-LM-Head-Weight"></a>
       - LM Head Weight Updating with AdamW  
         Hyper Param example:  
         > η = 0.001 (Learning rate)  
         β1 = 0.9 (Firt moment decay coefficient)  
         β2=0.999  (Second  moment decay coefficient)  
         ϵ=10<sup>-8</sup> (Numerical stability constant)  
         λ = 0.01(Weight decay coefficient)  

         First Moment m:  
         > $
           m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t  
           $  
         Shape of m is same as W<sub>LM</sub> = [V, d]

         Second Moment v:
         > $
           v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2
           $  
         Shape of v is same as W<sub>LM</sub> = [V, d]  

         > *Above caculation method is EMA: Exponential Moving Average,  instead of arithmetic mean*  

         Since *m* intiliazed with 0, caused the m<sub>t</sub> is too small at the begining, for example:  
         > Step 1: m<sub>1</sub>=0.9×0+0.1×g<sub>1</sub>=0.1g<sub>1</sub>  
         ​Step 2: m<sub>2</sub>=0.9×0.1g<sub>1</sub>+0.1×g<sub>2</sub>≈0.19g  

         So here's the adjustment formular for m and v:  
         > $  
         \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
         $  
         $
         \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
         $  

         Weight updating:    

         > $W_t = W_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot W_{t-1} \right)$  

           Everytime $\lambda$ pull Weight to 0 a little bit, it affects Weight straightly in AdamW, insteadly it affects Gradiant in Adam, that's the major different between AdamW and Adam.  
           *(It's possible to skip the LM Head Weight updating by setting to FALSE)*  
           <br>  

      - $\frac{\partial L}{\partial H_{norm}}$ is for passing loss to transformer, the shape is $[B, T, d]$   
      <br>  

    <a href="" id="Backward-RMSNorm"></a>
2. **Final RMSNorm**   
    - Gradiant to $γ \in R^{(d)}$ (Scale Param)  
          > $
          \frac{\partial \mathcal{L}}{\partial \gamma_i} = \frac{\partial \mathcal{L}}{\partial H_{norm}} \cdot \hat{x}_i
          $    
          $\hat{x}$ is output of standalization of final RMSNorm, the Shape of $\hat{x}$ is [B,T,d]  
          $\frac{\partial \mathcal{L}}{\partial \gamma_i}$ Shape is [d], it's from Sum operation(Compare to the Broadcast in Forward Pass)  


    - Gradiant to $β \in R^{(d)}$ (Shift Param - only used in LayerNorm)  
      $
      \frac{\partial \mathcal{L}}{\partial \beta_i} = \frac{\partial \mathcal{L}}{\partial H_{norm}}
      $  
    - Global L2 Norm Clipping  
      γ and β will not lead the Clipping because they only have [d] elements.  
        $\|g\|_{global} = \sqrt{\|g_{W_{head}}\|_2^2 + \|g_{\gamma}\|_2^2 + \|g_{body}\|_2^2 + \cdots}$  
    - AdamW  
      > $\gamma_t = \gamma_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}  \right)$  
      m and v need to be FP32

      For γ, we will not perform Weight decay, that's why there's no $\lambda\cdot\gamma_{t-1}$ in above formular, because pull γ to 0 will block the gradiant(?).  

      RMSNorm will not use β, it will follow the same approach as γ in layerNorm.  
    - Gradiant to $x \in R^{(B,T,d)}$, x is the output of last layer.    
    $$
    \frac{\partial \mathcal{L}}{\partial h^N_{ij}} = \frac{1}{\text{RMS}(h_i)} \left( g_{ij} - \frac{\hat{h}_{ij}}{d} \sum_{k=1}^{d} g_{ik} \hat{h}_{ik} \right)
    $$  
      <br>  
3. **FFN + RMSNorm**   
    - $W_{down}$  
      > $
      \frac{\partial \mathcal{L}}{\partial W_{down}} = \frac{\partial \mathcal{L}}{\partial h^N} \cdot act^\top
      $  

      $act => activation \in R^{L \times d}$  
      $\frac{\partial \mathcal{L}}{\partial h^N}$ is from above Final RMSNorm's output  
      
      Shape of $\frac{\partial \mathcal{L}}{\partial W_{down}}$ is [d, d<sub>ff</sub>]

      For update the W<sub>down</sub>:  
      - Global L2 Norm Clipping  
        $
        \|g\|_{global} = \sqrt{\|g_{W_{down}}\|_2^2 + \|g_{W_{gate}}\|_2^2 + \|g_{W_{up}}\|_2^2 + \|g_{W_{head}}\|_2^2 + \cdots}
        $
      - AdamW  
        HyperParam example:
        > η = same as the attention's  
        λ =0.1 (Weight Decay)  
        β<sub>1</sub> = 0.9  
        β<sub>2</sub> = 0.95  
        ϵ = 1e-8  

        Weight updating process is same as [LM Head Weight's updating](#Backward-LM-Head-Weight)    
        <br>
      > $
      \frac{\partial \mathcal{L}}{\partial act} = W_{down}^\top \cdot \frac{\partial \mathcal{L}}{\partial h^N}
      $  

      Shape of $\frac{\partial \mathcal{L}}{\partial act}$ is [B, T, d<sub>ff</sub>]  
      $\frac{\partial \mathcal{L}}{\partial act}$ is for passing the gradiant to SiLU(gate)⊙up

    - SiLU(gate)⊙up  
      $
      \frac{\partial \mathcal{L}}{\partial \text{SiLU}(gate)} = \frac{\partial \mathcal{L}}{\partial act} \odot up
      $  

      $
      \frac{\partial \mathcal{L}}{\partial up} = \frac{\partial \mathcal{L}}{\partial act} \odot \text{SiLU}(gate)
      $  
      Refer to the [Forward Pass](#swiglu) for these 2 branches  

    - Up and W<sub>up</sub>  
      > $
      \frac{\partial \mathcal{L}}{\partial h_{norm}^{up}} = W_{up}^\top \cdot \frac{\partial \mathcal{L}}{\partial up}
      $  
      For passing backward, will plus gate's grandiant  

      > $
      \frac{\partial \mathcal{L}}{\partial W_{up}} = \frac{\partial \mathcal{L}}{\partial up} \cdot h_{norm}^\top
      $  
      For updating the W<sub>up</sub>, the updating process is exactly same as W<sub>down</sub>

    - SiLU(gate) and W<sub>gate</sub>  
      > $
      \frac{\partial \mathcal{L}}{\partial gate} = \frac{\partial \mathcal{L}}{\partial \text{SiLU}(gate)} \odot \text{SiLU}'(gate)
      $  
      SiLU′(z)=σ(z)+z⋅σ(z)⋅(1−σ(z))=σ(z)⋅(1+z⋅(1−σ(z)))  


      > $
      \frac{\partial \mathcal{L}}{\partial h_{norm}^{gate}} = W_{gate}^\top \cdot \frac{\partial \mathcal{L}}{\partial gate}
      $  


      > $
      \frac{\partial \mathcal{L}}{\partial W_{gate}} = \frac{\partial \mathcal{L}}{\partial gate} \cdot h_{norm}^\top
      $  

      W<sub>gate</sub> updating process is same as [LM Head Weight's updating](#Backward-LM-Head-Weight) 
    - Sum of Up and Gate's grandiant as input of RMSNorm2's backpropagation  
      $
      \frac{\partial \mathcal{L}}{\partial h_{norm}} = \frac{\partial \mathcal{L}}{\partial h_{norm}}\bigg|_{gate} + \frac{\partial \mathcal{L}}{\partial h_{norm}}\bigg|_{up}
      $  
      $\frac{\partial \mathcal{L}}{\partial h_{norm}} \in \mathbb R^{[B, T, d]}$  
    - RMSNorm  
      > $
      \frac{\partial \mathcal{L}}{\partial \gamma_{2,i}} = \sum_{b,t} \frac{\partial \mathcal{L}}{\partial h_{norm,b,t,i}} \cdot \hat{x}_{b,t,i}
      $  

      $\frac{\partial \mathcal{L}}{\partial \gamma_{2}} \in \mathbb R ^{[d]}$ compressioned from [B,T,d] to [d] by sum on B and T dimention.  
      Update $\gamma_{2}$ by AdamW, same process as the [Final RMSNorm](#Backward-RMSNorm)  

      > $
      \frac{\partial \mathcal{L}}{\partial h_{mid,i}} = \frac{\gamma_{2,i}}{r} \left[ \frac{\partial \mathcal{L}}{\partial h_{norm,i}} - \frac{h_{mid,i}}{d \cdot r^2} \sum_{j=1}^d \frac{\partial \mathcal{L}}{\partial h_{norm,j}} \cdot h_{mid,j} \right]
      $  

      $
      \frac{\partial \mathcal{L}}{\partial h_{mid}} \in \mathbb R^{B, T, d}$  
      r=RMS(h<sub>mid</sub>)  

      <br>  

4. **1st Gradient Accumulation**  
    > $
    \frac{\partial \mathcal{L}}{\partial h_{mid}}^{total} = \frac{\partial \mathcal{L}}{\partial h_N} + \frac{\partial \mathcal{L}}{\partial h_{mid}}\bigg|_{FFN}
    $  

    $\frac{\partial \mathcal{L}}{\partial h_{mid}} \in \mathbb R ^{[B, T, d]}$ as the input of Attention's backpropagation.  
      <br>  

5. **Attention + RMSNorm**  
    - W<sub>O</sub>  
      > $
      \frac{\partial \mathcal{L}}{\partial W_O} = \left(\frac{\partial \mathcal{L}}{\partial h_{mid}}\right)^\top \cdot out\_attn
      $  

      $\frac{\partial \mathcal{L}}{\partial W_O}\in \mathbb R^{[d,d]}$ and updating process is same as [LM Head Weight's updating](#Backward-LM-Head-Weight) 

      > $
      \frac{\partial \mathcal{L}}{\partial out\_attn} = W_O^\top \cdot \frac{\partial \mathcal{L}}{\partial h_{mid}}
      $  
      $\frac{\partial \mathcal{L}}{\partial out\_attn} \in \mathbb R^{[B, T, d]}$


    - V & attn  
      $
      \frac{\partial \mathcal{L}}{\partial V} = attn^\top \cdot \frac{\partial \mathcal{L}}{\partial out\_attn}
      $  
      $
      \frac{\partial \mathcal{L}}{\partial attn} = \frac{\partial \mathcal{L}}{\partial out\_attn} \cdot V^\top
      $  
    - Softmax  
      $
      \frac{\partial \mathcal{L}}{\partial scores} = attn \odot \left( \frac{\partial \mathcal{L}}{\partial attn} - \text{rowsum}\left(\frac{\partial \mathcal{L}}{\partial attn} \odot attn\right) \right)
      $  

    - Q<sub>rot</sub> & K<sub>rot</sub>  
      > $
      \frac{\partial \mathcal{L}}{\partial Q_{rot}} = \frac{\partial \mathcal{L}}{\partial scores} \cdot K_{rot} / \sqrt{d_k}
      $  

      > $
      \frac{\partial \mathcal{L}}{\partial K_{rot}} = \left(\frac{\partial \mathcal{L}}{\partial scores}\right)^\top \cdot Q_{rot} / \sqrt{d_k}
      $  
    - RoPE  
      > $
      \frac{\partial \mathcal{L}}{\partial Q} = \text{RoPE}^{-1}\left(\frac{\partial \mathcal{L}}{\partial Q_{rot}}, pos\right)
      $  

      > $
      \frac{\partial \mathcal{L}}{\partial K} = \text{RoPE}^{-1}\left(\frac{\partial \mathcal{L}}{\partial K_{rot}}, pos\right)
      $  

    - Q & K & V  
      > $
      \frac{\partial \mathcal{L}}{\partial x\_norm} = W_Q^\top \cdot \frac{\partial \mathcal{L}}{\partial Q} + W_K^\top \cdot \frac{\partial \mathcal{L}}{\partial K} + W_V^\top \cdot \frac{\partial \mathcal{L}}{\partial V}
      $  

      $\frac{\partial \mathcal{L}}{\partial x\_norm} \in \mathbb R^{[B, T, d]}$

      > $
      \frac{\partial \mathcal{L}}{\partial W_Q} = \left(\frac{\partial \mathcal{L}}{\partial Q}\right)^\top \cdot x\_norm
      $  
      $
      \frac{\partial \mathcal{L}}{\partial W_K} = \left(\frac{\partial \mathcal{L}}{\partial K}\right)^\top \cdot x\_norm
      $  
      $
      \frac{\partial \mathcal{L}}{\partial W_V} = \left(\frac{\partial \mathcal{L}}{\partial V}\right)^\top \cdot x\_norm
      $  

      $\frac{\partial \mathcal{L}}{\partial W_Q},\frac{\partial \mathcal{L}}{\partial W_K},\frac{\partial \mathcal{L}}{\partial W_V} \in \mathbb R^{[d,d]}$
    - RMSNorm
  

  <a href="" id="whereami"></a>  