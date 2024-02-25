import os
import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, classification_report

import transformers
import torch, torchaudio, torchtext
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from transformers import BertTokenizer, BertModel, AutoModel, AutoProcessor
from tqdm import tqdm

from utils.CustomTransformer import CustomEncoder, PositionalEncoding, LayerNorm, PositionwiseFeedForward
from utils.constants import *
from utils.data_utils import *
from utils.train_utils import *

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.4f}'.format


def main():
    train_df, val_df, test_df = load_and_split_df(DF_PATH)
    class_weight = compute_class_weight(train_df)

    tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_CARD)
    embedder = BertModel.from_pretrained(TEXT_MODEL_CARD).to(device)

    for params in embedder.parameters():
        params.requires_grad = EMBEDDING_TRAINABLE
    
    train_dataset, val_dataset, test_dataset = load_datasets(train_df, val_df, test_df)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    gc.collect()
    
    train_evaluate_all_models(
        class_weight, 
        train_dataloader, 
        val_dataloader, 
        test_dataloader,
        tokenizer, 
        embedder)

if __name__ == '__main__':
    main()