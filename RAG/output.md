# TEST 1

jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "How many layers are in the encoder?" 
Uninstalled 1 package in 0.71ms
Installed 1 package in 2ms

--- score 0.587 | 1706.03762v7.pdf p.3 | 3.1 Encoder and Decoder Stacks
3.1 Encoder and Decoder Stacks
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed bylayer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the e

--- score 0.458 | 1706.03762v7.pdf p.5 | 3.3 Position-wise Feed-Forward Networks
3.3 Position-wise Feed-Forward Networks
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
<!-- formula-not-decoded -->
While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and

--- score 0.432 | 1706.03762v7.pdf p.5 | 3.2.3 Applications of Attention in our Model
3.2.3 Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms insequence-to-sequence models such as [38, 2, 9].
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the s

--- score 0.418 | 1706.03762v7.pdf p.2, 3 | 3 Model Architecture
3 Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.
Figure 1: The Transformer - model architecture.
The Transformer follows this overall architectur

--- score 0.404 | 1706.03762v7.pdf p.1 | Abstract
Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solelyon attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BL
jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "What BLEU score did the big Transformer get on English-to-German?" 
Uninstalled 1 package in 0.41ms
Installed 1 package in 1ms

--- score 0.507 | 1706.03762v7.pdf p.8 | 6.1 Machine Translation
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.
On the WMT 2014 English-to-French translation task, our big 

--- score 0.413 | 1706.03762v7.pdf p.1 | Abstract
Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solelyon attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BL

--- score 0.397 | 1706.03762v7.pdf p.9 | 6.2 Model Variations
6.2 Model Variations
Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

base, N = 6. base, dmodel = 512. base, dff = 2048. base, h = 8. base, dk = 64. base, dv = 64. base, Pdrop = 0.1. base, ϵls = 0.1. base, train steps = 100K. base, PPL (dev) = 4.92. base, BLEU (dev) = 25.8. base, params ×106 = 65. , N = . , dmodel = . , df

--- score 0.394 | 1706.03762v7.pdf p.8, 9 | 6.2 Model Variations
6.2 Model Variations
To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.
5 We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.

--- score 0.379 | 1706.03762v7.pdf p.9, 10 | 6.3 English Constituency Parsing
6.3 English Constituency Parsing
To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not beenable to attain state-of-the-art results in small-data regimes [37].
We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-sup
jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "How many attention heads does the base model use, and what is the key dimension?" 
Uninstalled 1 package in 0.35ms
Installed 1 package in 1ms

--- score 0.483 | 1706.03762v7.pdf p.4, 5 | 3.2.2 Multi-Head Attention
3.2.2 Multi-Head Attention
Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
4 To illustrate why the dot products get large,

--- score 0.456 | 1706.03762v7.pdf p.9 | 6.2 Model Variations
6.2 Model Variations
In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as e

--- score 0.442 | 1706.03762v7.pdf p.5 | 3.2.3 Applications of Attention in our Model
3.2.3 Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms insequence-to-sequence models such as [38, 2, 9].
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the s

--- score 0.420 | 1706.03762v7.pdf p.2 | 2 Background
2 Background
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number ofoperations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Tr

--- score 0.417 | 1706.03762v7.pdf p.1 | Abstract
Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solelyon attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BL
jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "What optimizer and warmup steps were used?" 
Uninstalled 1 package in 0.53ms
Installed 1 package in 1ms

--- score 0.503 | 1706.03762v7.pdf p.7 | 5.3 Optimizer
5.3 Optimizer
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10-9 . We varied the learning rate over the course of training, according to the formula:
<!-- formula-not-decoded -->
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.

--- score 0.376 | 1706.03762v7.pdf p.7, 8 | 5.4 Regularization
5.4 Regularization
We employ three types of regularization during training:
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

ByteNet [18], BLEU.EN-DE = 23.75. ByteNet [18], BLEU.EN-FR = . ByteNet [18], Training Cost (FLOPs).EN-DE = . ByteNet [18], Training Cost (FLOPs).EN-FR = . Deep-Att + PosUnk [39], BLEU.EN-DE = . Deep-Att + PosUnk [39], BLEU.EN-FR = 39.2. Deep-Att + PosUnk [39], Training Cost (FLOPs).EN-DE = . Deep-Att + PosUnk [39],Training Cost

--- score 0.374 | 1706.03762v7.pdf p.7 | 5.2 Hardware and Schedule
5.2 Hardware and Schedule
We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

--- score 0.347 | 1706.03762v7.pdf p.8 | 6.1 Machine Translation
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.
On the WMT 2014 English-to-French translation task, our big 

--- score 0.322 | 1706.03762v7.pdf p.9, 10 | 6.3 English Constituency Parsing
6.3 English Constituency Parsing
To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not beenable to attain state-of-the-art results in small-data regimes [37].
We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-sup
jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "How long did the big model take to train, and on what hardware?" 
Uninstalled 1 package in 0.59ms
Installed 1 package in 2ms

--- score 0.539 | 1706.03762v7.pdf p.7 | 5.2 Hardware and Schedule
5.2 Hardware and Schedule
We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

--- score 0.493 | 1706.03762v7.pdf p.8 | 6.1 Machine Translation
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.
On the WMT 2014 English-to-French translation task, our big 

--- score 0.387 | 1706.03762v7.pdf p.9, 10 | 6.3 English Constituency Parsing
6.3 English Constituency Parsing
To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not beenable to attain state-of-the-art results in small-data regimes [37].
We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-sup

--- score 0.383 | 1706.03762v7.pdf p.9 | 6.2 Model Variations
6.2 Model Variations
Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

base, N = 6. base, dmodel = 512. base, dff = 2048. base, h = 8. base, dk = 64. base, dv = 64. base, Pdrop = 0.1. base, ϵls = 0.1. base, train steps = 100K. base, PPL (dev) = 4.92. base, BLEU (dev) = 25.8. base, params ×106 = 65. , N = . , dmodel = . , df

--- score 0.370 | 1706.03762v7.pdf p.7 | 5 Training
5 Training
This section describes the training regime for our models.
jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "What does Figure 1 show?" 
Uninstalled 1 package in 0.61ms
Installed 1 package in 2ms

--- score 0.356 | 1706.03762v7.pdf p.2, 3 | 3 Model Architecture
3 Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.
Figure 1: The Transformer - model architecture.
The Transformer follows this overall architectur

--- score 0.346 | 1706.03762v7.pdf p.14, 15 | Input-Input Layer5
Input-Input Layer5
Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'its' for attention heads 5 and 6. Note that the attentions are very sharp for this word.
Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.

--- score 0.338 | 1706.03762v7.pdf p.1 | 
Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

--- score 0.330 | 1706.03762v7.pdf p.9 | 6.2 Model Variations
6.2 Model Variations
In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as e

--- score 0.328 | 1706.03762v7.pdf p.3 | 3.1 Encoder and Decoder Stacks
3.1 Encoder and Decoder Stacks
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed bylayer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the e
jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "What is the formula for scaled dot-product attention?" 
Uninstalled 1 package in 0.35ms
Installed 1 package in 2ms

--- score 0.595 | 1706.03762v7.pdf p.4 | 3.2.1 Scaled Dot-Product Attention
3.2.1 Scaled Dot-Product Attention
We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values.
In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:
<!-- formula-not-decoded -->
The two most

--- score 0.522 | 1706.03762v7.pdf p.4 | Scaled Dot-Product Attention
Scaled Dot-Product Attention
Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

--- score 0.480 | 1706.03762v7.pdf p.4, 5 | 3.2.2 Multi-Head Attention
3.2.2 Multi-Head Attention
Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
4 To illustrate why the dot products get large,

--- score 0.435 | 1706.03762v7.pdf p.3 | 3.2 Attention
3.2 Attention
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum

--- score 0.400 | 1706.03762v7.pdf p.9 | 6.2 Model Variations
6.2 Model Variations
In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as e
jhunt14@eccai00:~/backendRAGtesting$ uv run src/query.py "Who wrote “Layer normalization”?" 
Uninstalled 1 package in 0.38ms
Installed 1 package in 2ms

--- score 0.399 | 1706.03762v7.pdf p.10, 11 | References
References
- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
- [2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
- [3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.
- [4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.
- [5] Kyun

--- score 0.331 | 1706.03762v7.pdf p.7, 8 | 5.4 Regularization
5.4 Regularization
We employ three types of regularization during training:
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

ByteNet [18], BLEU.EN-DE = 23.75. ByteNet [18], BLEU.EN-FR = . ByteNet [18], Training Cost (FLOPs).EN-DE = . ByteNet [18], Training Cost (FLOPs).EN-FR = . Deep-Att + PosUnk [39], BLEU.EN-DE = . Deep-Att + PosUnk [39], BLEU.EN-FR = 39.2. Deep-Att + PosUnk [39], Training Cost (FLOPs).EN-DE = . Deep-Att + PosUnk [39],Training Cost

--- score 0.324 | 1706.03762v7.pdf p.3 | 3.1 Encoder and Decoder Stacks
3.1 Encoder and Decoder Stacks
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed bylayer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the e

--- score 0.318 | 1706.03762v7.pdf p.1 | Attention Is All You Need
Attention Is All You Need
Ashish Vaswani∗ Google Brain avaswani@google.com Noam Shazeer∗ Google Brain noam@google.com Niki Parmar∗ Google Research nikip@google.com Jakob Uszkoreit∗ Google Research usz@google.com
Llion Jones∗ Google Research llion@google.com Aidan N. Gomez∗ † University of Toronto aidan@cs.toronto.edu Łukasz Kaiser∗ Google Brain lukaszkaiser@google.com Illia Polosukhin∗ ‡ illia.polosukhin@gmail.com

--- score 0.318 | 1706.03762v7.pdf p.9, 10 | 6.3 English Constituency Parsing
6.3 English Constituency Parsing
To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not beenable to attain state-of-the-art results in small-data regimes [37].
We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-sup