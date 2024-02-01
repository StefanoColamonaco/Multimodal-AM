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
        <th colspan="1">Train</th>
        <th colspan="1">Test</th>
        <th colspan="1">Train</th>
        <th colspan="1">Test</th>
    </tr>
    <tr>
        <td colspan="1">unaligned</td>
        <td colspan="1">0.700</td>
        <td colspan="1">0.003</td>
        <td colspan="1">0.699</td>
        <td colspan="1">0.002</td>
    </tr>
    <tr>
        <td colspan="1">multimodal</td>
        <td colspan="1">0.699</td>
        <td colspan="1">0.005</td>
        <td colspan="1">0.698</td>
        <td colspan="1">0.006</td>
    </tr>
    <tr>
        <td colspan="1">ensembling</td>
        <td colspan="1">0.683</td>
        <td colspan="1">0.002</td>
        <td colspan="1">0.682</td>
        <td colspan="1">0.001</td>
    </tr>
    <tr>
        <td colspan="1">text-only</td>
        <td colspan="1">0.681</td>
        <td colspan="1">0.002</td>
        <td colspan="1">0.680</td>
        <td colspan="1">0.002</td>
    </tr>
    <tr>
        <td colspan="1">audio_only</td>
        <td colspan="1">0.549</td>
        <td colspan="1">0.002</td>
        <td colspan="1">0.523</td>
        <td colspan="1">0.016</td>
    </tr>
  </tbody>
</table>