import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the dataset
DF_PATH = 'multimodal-dataset/files/MM-USElecDeb60to16/MM-USElecDeb60to16.csv'
AUDIO_PATH = 'multimodal-dataset/files/MM-USElecDeb60to16/audio_clips'

SAMPLE_RATE = 16_000
DOWNSAMPLE_FACTOR = 1/100

BASE_OUTPUT_PATH = 'output_files'
if not os.path.exists(BASE_OUTPUT_PATH):
    os.makedirs(f'{BASE_OUTPUT_PATH}')
    os.makedirs(f'{BASE_OUTPUT_PATH}/datasets')
    os.makedirs(f'{BASE_OUTPUT_PATH}/weights')
    os.makedirs(f'{BASE_OUTPUT_PATH}/history')
    os.makedirs(f'{BASE_OUTPUT_PATH}/results')
    
LOAD_DATASET_PATH = f'{BASE_OUTPUT_PATH}/datasets'
WEIGHTS_PATH = f'{BASE_OUTPUT_PATH}/weights'
HISTORY_PATH = f'{BASE_OUTPUT_PATH}/history'
RESULTS_PATH = f'{BASE_OUTPUT_PATH}/results'


# Dataset params
REMOVE_OTHER = True
MODEL_NUM_LABELS = 2 if REMOVE_OTHER else 3
OTHER_LABEL = 'O'
OTHER_LABEL_ID = 2
    
# Model cards
TEXT_MODEL_CARD = 'bert-base-uncased'
AUDIO_MODEL_CARD = 'facebook/wav2vec2-base-960h'

# Helper dictionaries
LABEL_2_ID = {'Claim': 0, 'Premise': 1, 'O': 2}
ID_2_LABEL = {0: 'Claim', 1: 'Premise', 2: 'O'}

# Embedding params
EMBEDDING_TRAINABLE = False
EMBEDDING_DIM = 768
BATCH_SIZE = 12

MODEL_NAMES = ['Text-Only', 'Audio-Only', 'CSA', 'Ensemble', 'Mul-TA']
SEEDS = [1, 42, 69, 100, 420]

# Training params
EPOCHS = 10
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-3
LR_DECAY_FACTOR = 1e-1
LR_DECAY_PATIENCE = 3
VERBOSE_TRAIN = True
DEBUG_TRAIN = False


# Models Hyperparameters
### shared
HEAD_HIDDEN_DIMENSION = 256
DROPOUT_PROB = 0.1
HIDDEN_STATE_INDEX = 8
### Audio-Only
AUDIO_ONLY_N_HEADS = 8
AUDIO_ONLY_D_FFN = 100
AUDIO_ONLY_N_LAYERS = 1
### CSA
CSA_N_HEADS = 4
CSA_D_FFN = 2048
CSA_N_LAYERS = 1
### Ensembling
ENSEMBLING_N_HEADS = 4
ENSEMBLING_D_FFN = 2048
ENSEMBLING_N_LAYERS = 1
### Mul-TA
MULTA_N_BLOCKS = 4
MULTA_N_HEADS = 4
MULTA_D_FFN = 2048
