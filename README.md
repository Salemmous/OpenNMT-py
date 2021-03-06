# OpenNMT-py: neural-interlingua branch 
[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)  

## About this branch
`neural-interlingua` is an implementation of an Attention Bridge architecture to obtain the language independent sentence representation. The proposed architecture is a multilingual implementation of the conjugate attention encoder-decoder NMT ([Cífka and Bojar, 2018](https://arxiv.org/pdf/1805.06536.pdf)), trainable with a language rotating schedule. In other words, we propose to simultaneously use the following two 

[fig1]: https://github.com/Helsinki-NLP/OpenNMT-py/blob/ATT-ATT/att-att.png "compound attention Figure 1"
[fig2]: https://github.com/Helsinki-NLP/OpenNMT-py/blob/neural-interlingua/data/multi_enc-dec_diagram.png "multiple encoders & decoders Figure 2"



|    Cífka and Bojar's Compound Attention  | Language rotation  |
|:----------------: |:----------------: |
| ![alt text][fig1] | ![alt text][fig2] |
|                   |                   |

Please refer to the preprint [Multilingual NMT with a language-independent attention bridge](https://arxiv.org/pdf/1811.00498.pdf) for further details.

#### Requirements:
- same as OpenNMT-py (see [Installation](http://opennmt.net/OpenNMT-py/main.html#installation))
- for installing in [CSC-TAITO](https://www.csc.fi/) please feel free to use the script `attBridge_setup.sh` (notice that you will need to modify the paths in it)
- Right now it supports [OpenNMT-py0.2.1](https://github.com/OpenNMT/OpenNMT-py/releases/tag/0.2.1) which has full compatibility with: 
  ```
  - python >= 3.6 < 3.7
  - pytorch >= 0.4.0 < 1.0
  - torchtext >= 0.3.0
  ```
 - We have kept the documentation of the `master` branch for OpenNMT-py0.2.1 [below on this README](#opennmt-py-open-source-neural-machine-translation). Please follow it for version compatibility of libraries.
 
#### USAGE:
##### Training
Same as OpenNMT-py (see [Full documentation](http://opennmt.net/OpenNMT-py/) ) with main differences in the following **NEW** flags to include multilingual models for `train.py` routine: 
 - `-data` (str) [] Give a path to the (preprocessed) data for each language-pair to train, separate them by a space
 - `-src-tgt` (str) [] Indicate the language pairs to train separated by a space (must contain the same number of pairs as the paths given through the `-data` flag). _Example: en-fr de-cs cs-fr_ 
 - `-use_attention_bridge` () [] Indicates weather to use the compound attention model. Including it will  set to `True` the useage of the self-attention layer
 - `-attention_heads` (int) [default=4] Indicates the number of attention heads (columns) to use on the attention bridge; the `r` parameter from Cífka and Bojar (2018)
 - `-init_decoder` (str) [default = 'rnn_final_state'] Choose a method to initialize decoder. With the final state
                       of the decoder (as in Cifka and Bojar (2018)) or with the average of the heads of the attention bridge (as in [Sennrich et al. (2017)](https://arxiv.org/pdf/1703.04357.pdf) ).
                       choices=['rnn_final_state', 'attention_matrix']
 - `-report_bleu` () [] Using this flag will make the system print the BLEU scores obtained during validation.

##### Translating
To translate using a multilingual model use `translate_multimodel.py` instead of  `translate.py`. This routine is called similarly to `train.py` (see OpenNMT-py documentation), with the following **NEW** flags:
 - `-src_lang` (str) [] source lang of the data to be translated
 - `-tgt_lang` (str) [] target language of the translation
 
### Example:
1- Get some data and **preprocess** it by running the script `data/prep-data.sh`. 
   NOTE: if you ran the `attBridge_setup.sh` script (for installing on CSC), this has already been done.
   ```
   ONMT=<where/you/cloned/this/repo/OpenNMT-py>
   cd ${ONMT}/data
   source ./prep-data.sh
   cd ${ONMT}
   ```
2- **Train** a multilingual model with 10 language pairs (we will use zero-shot translation for `cs <-> en`)
   ```
   SAVE_PATH=$ONMT/models/demo
   mkdir -p $SAVE_PATH
   python train.py -data data/sample_data/de-cs/data \
                      data/sample_data/fr-cs/data \
                      data/sample_data/de-en/data \
                      data/sample_data/fr-en/data \
                      data/sample_data/cs-de/data \
                      data/sample_data/en-de/data \
                      data/sample_data/fr-de/data \
                      data/sample_data/cs-fr/data \
                      data/sample_data/de-fr/data \
                      data/sample_data/en-fr/data \
                -src_tgt de-cs fr-cs de-en fr-en cs-de en-de fr-de cs-fr de-fr en-fr \
                -save_model ${SAVE_PATH}/MULTILINGUAL          \
                -use_attention_bridge \
                -attention_heads 20 \
                -rnn_size 512 \
                -rnn_type GRU \
                -encoder_type brnn \
                -decoder_type rnn \
                -enc_layers 2 \
                -dec_layers 2 \
                -word_vec_size 512 \
                -global_attention mlp \
                -train_steps 100000 \
                -valid_steps 10000 \
                -optim adam \
                -learning_rate 0.0002 \
                -batch_size 256 \
                -gpuid 0 \
                -save_checkpoint_steps 10000
   ```  

3- **Translate** from src to tgt. You can see that even though we did not train `en-cs` or `cs-en` you can choose this language pair.
   ```
   for src in de en; do
     for tgt in fr cs; do
       python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_step_1000.pt \
            -src_lang ${src} \
            -src data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.${src} \
            -tgt_lang ${tgt} \
            -tgt data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.${tgt} \
            -report_bleu \
            -gpu 0 \
            -use_attention_bridge \
            -output ${SAVE_PATH}/MULTILINGUAL_prediction_${src}-${tgt}.txt \
            -verbose
     done
   done
   ```
#### Notes on the Example:

On the previous example we are using the attention bridge to obtain the language independent sentence representation by performing MT. We use all possible combinations of the available languages of the [multi30k dataset](https://github.com/multi30k/dataset) with the following specifications.

 - **ENCODERS:** For each encoder (the `de` and `en` encoders in this case), we use 
    1. a 2-layered biGRU; and  
 
    We adapted the Self-attentive sentence embedding proposed by [Lin et al. (2017)](https://arxiv.org/pdf/1703.03130.pdf) for being use in the OpenNMT environment. Some code is provided in [their GitHub](https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py)
    
 - **ATTENTION BRIDGE:**
   -  A self-attention layer with 20 `attention_heads`, shared among all encoders and decoders.
   
 - **DECODERS:** For each decoder (`fr` and `cs` in this case) we use a modified version of the OpenNMT `InputFeedRNNDecoder` from `onmt/Models.py`, i.e.,  
    1. a traditional Bahdanau attention layer (1 attention head)
    2. a 2-layered unidirectional GRU. If `-init_decoder` is set to `attention_matrix`,  we set `s_0`, the initial the decoder state, as the average of the columns of the attention bridge - in a similar fashion as [Sennrich et al.(2017)](https://arxiv.org/pdf/1703.04357.pdf). 
 
    i.e., '''
s_0 = tanh(W_init * h_avrg);
'''
    where h_avrg is the average of the sentence embedding, M, (instead of taking the average over the hidden states of the RNN, as in Sennrich(2017))


### BibTeX:
preprint of technical report:
```
@article{attention_bridge,
  author    = {Ra{\'{u}}l V{\'{a}}zquez and
               Alessandro Raganato and
               J{\"{o}}rg Tiedemann and
               Mathias Creutz},
  title     = {Multilingual {NMT} with a language-independent attention bridge},
  journal   = {CoRR},
  volume    = {abs/1811.00498},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.00498},
  archivePrefix = {arXiv},
  eprint    = {1811.00498}
}

```



------------------
OpenNMT-py0.2.1 README:
------------------
# OpenNMT-py: Open-Source Neural Machine Translation
====================================================
[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)
[![Run on FH](https://img.shields.io/badge/Run%20on-FloydHub-blue.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains.

Codebase is relatively stable, but PyTorch is still evolving. We currently only support PyTorch 0.4 and recommend forking if you need to have stable code.

OpenNMT-py is run as a collaborative open-source project. It is maintained by [Sasha Rush](http://github.com/srush) (Cambridge, MA), [Ben Peters](http://github.com/bpopeters) (Lisbon), and [Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai). The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC). 
We love contributions. Please consult the Issues page for any [Contributions Welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tagged post. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>


Table of Contents
=================
  * [Full Documentation](http://opennmt.net/OpenNMT-py/)
  * [Requirements](#requirements)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Run on FloydHub](#run-on-floydhub)
  * [Citation](#citation)

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Note that we currently only support PyTorch 0.4.

## Features

The following OpenNMT features are implemented:

- [data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- [Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [TensorBoard](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)
- ["Attention is all you need"](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Multi-GPU](http://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)
- Inference time loss functions.

Beta Features (committed):
- Structured attention
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpuid 1` to use (say) GPU 1.

### Step 3: Translate

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Alternative: Run on FloydHub

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

Click this button to open a Workspace on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=opennmt-py&utm_campaign=jul_2018) for training/testing your code.


## Pretrained embeddings (e.g. GloVe)

Go to tutorial: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://forum.opennmt.net/t/how-to-use-glove-pre-trained-embeddings-in-opennmt-py/1011)

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py.

http://opennmt.net/Models-py/



## Citation

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)


[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
