# How to Train Your Transformer from Scratch for Machine Learning Translation

In this blog, I will use the **BERT** architecture to perform translation from *Portuguese* to *English*. This blog provides a *comprehensive* guide to the entire training pipeline, encompassing *Data Preprocessing, Building the Model, Training the BERT Model, and Exporting the Trained Model.* The best aspect of this blog is that I have ensured to elucidate each code section thoroughly.

## A Brief Overview of the Transformer Architecture
*Transformer* has emerged as a prominent subject of research in the realm of AI, particularly in *NLP*, since its introduction in the groundbreaking paper titled *Attention is All You Need*.

<div align="center">
    <img src="images/transformer.png" alt="Transformer Architecture" width="300">
    <br>
    Transformer Architecture.
</div>
One crucial component of the transformer is the **Multi-Head Attention** mechanism. Let's take a moment to understand its theoretical aspects.

### Multi-Head Attention(*MHA*):
Before diving into *MHA* mechanism, we need to understand what does attention mean. 
<!-- What is attention in general, attention in NLP using examples,what are queries, keys, and values. Descibe the mathematical aspects. In encoder and in decoder how it differ,
talk about look ahead mask. Next time -->

<!-- While this blog won't delve deeply into the theoretical aspects of every transformer component, it will offer a detailed explanation of how to implement these components in the code section, allowing us to construct the transformer model from scratch. -->

*Credit: Tensorflow*
