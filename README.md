# Multimodal Argumentation Mining
This repository contains the code for the project "Multimodal Argumentation Mining" for the course "Natural Language Processing" at the University of Bologna.

## Project description
The goal of this project is to develop a multimodal argumentation mining system that is able to classify sentences belonging to presidential debates as <em>Claim</em> or <em>Premise</em>.

## Dataset
The dataset used for this project is the [MM-USElecDeb60to16](https://github.com/federicoruggeri/multimodal-am/tree/main/multimodal-dataset) presented in the paper [Multimodal Argument Mining: A Case Study in Political Debates](https://aclanthology.org/2022.argmining-1.15.pdf) by Mancini et al. (2022).

To generate the dataset, it's sufficient to run the script `audio_pipeline.py` in the folder `multimodal-dataset`. This script will download the audio files associated with the dataset, aligned with the texts at sentence level.

## Model
We implemented and tested five different models:
- a text-only model, based on a BERT encoder;
- an audio-only model, based on a Wav2Vec encoder;
- a Multi-modal Transformer model based on the concatenation of text and audio embeddings with a self-attention layer as presented in [Yu et al. (2023)](https://arxiv.org/pdf/2305.11579v2.pdf);
- an Ensemble model that aggregates the outputs of the text-only and audio-only models; 
- a Cross-modal Transformer model with cross-modal attention as presented in [Tsai et al. (2019)](https://arxiv.org/pdf/1906.00295.pdf).

## Code
The code is organized as follows:
- `CustomTransformer.py` contains the implementation of a custom Encoder Layer for the Multi-modal Transformer model;
- `main.ipynb` contains the code for the definition of the models, the training and, evaluation and the error analysis;
The code is available and ready to run at the following [Kaggle link](https://www.kaggle.com/andreazecca/multimodal-am)

## Results
The results are reporte in the following table:
<table>
  <tbody>
    <tr>
        <td colspan="1"><b></b></td>
        <td colspan="2"><b>Accuracy</b></td>
        <td colspan="2"><b>F1</b></td>
    </tr>
    <tr>
        <th colspan="1">Model</th>
        <th colspan="1">Mean</th>
        <th colspan="1">STD</th>
        <th colspan="1">Mean</th>
        <th colspan="1">STD</th>
    </tr>
    <tr>
        <td colspan="1">MulT-TA</td>
        <td colspan="1">0.7008</td>
        <td colspan="1">0.0040</td>
        <td colspan="1">0.7000</td>
        <td colspan="1">0.0038</td>
    </tr>
    <tr>
        <td colspan="1">CSA</td>
        <td colspan="1">0.6989</td>
        <td colspan="1">0.0055</td>
        <td colspan="1">0.6980</td>
        <td colspan="1">0.0056</td>
    </tr>
    <tr>
        <td colspan="1">Ensembling</td>
        <td colspan="1">0.6826</td>
        <td colspan="1">0.0033</td>
        <td colspan="1">0.6814</td>
        <td colspan="1">0.0034</td>
    </tr>
    <tr>
        <td colspan="1">Text-Only</td>
        <td colspan="1">0.6810</td>
        <td colspan="1">0.0018</td>
        <td colspan="1">0.6792</td>
        <td colspan="1">0.0021</td>
    </tr>
    <tr>
        <td colspan="1">Audio-Only</td>
        <td colspan="1">0.5472</td>
        <td colspan="1">0.0014</td>
        <td colspan="1">0.5264</td>
        <td colspan="1">0.0120</td>
    </tr>
  </tbody>
</table>