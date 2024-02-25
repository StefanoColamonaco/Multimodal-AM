import pandas as pd
import torch
from tqdm import tqdm
import os
import torchaudio
from transformers import AutoProcessor, AutoModel
from torch.nn.utils.rnn import pad_sequence

from utils.constants import *

def load_and_split_df(path, dataset_ratio=0.05):
    df = pd.read_csv(path, index_col=0)
    df = df[df['NewBegin'] != df['NewEnd']]
    if REMOVE_OTHER:
        df = df[df['Component'] != OTHER_LABEL]

    train_df = df[df['Set'] == 'TRAIN']
    val_df = df[df['Set'] == 'VALIDATION']
    test_df = df[df['Set'] == 'TEST']

    if dataset_ratio < 1:
        train_df = train_df.iloc[:int(dataset_ratio * len(train_df))]
        val_df = val_df.iloc[:int(dataset_ratio * len(val_df))]
        test_df = test_df.iloc[:int(dataset_ratio * len(test_df))]

    return train_df, val_df, test_df


def compute_class_weight(train_df):
    num_claim = len(train_df[train_df['Component'] == 'Claim'])
    num_premise = len(train_df[train_df['Component'] == 'Premise'])
    if not REMOVE_OTHER:
        num_other = len(train_df[train_df['Component'] == 'O'])
    if MODEL_NUM_LABELS == 2:
        claim_ratio = num_claim / (num_claim + num_premise)
        premise_ratio = num_premise / (num_claim + num_premise)
        weight = torch.tensor([1/(2*claim_ratio), 1/(2*premise_ratio)]).to(device)
    else:
        claim_ratio = num_claim / (num_claim + num_premise + num_other)
        premise_ratio = num_premise / (num_claim + num_premise + num_other)
        other_ratio = num_other / (num_claim + num_premise + num_other)
        weight = torch.tensor([1/(3*claim_ratio), 1/(3*premise_ratio), 1/(3*other_ratio)]).to(device)
    return weight


class MM_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for multimodal dataset
    """
    def __init__(self, df, audio_dir, sample_rate):
        """
        Args:
            df: dataframe containing the dataset
            audio_dir: directory containing the audio clips
            sample_rate: sample rate to use for audio clips
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate

        self.audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL_CARD)
        self.audio_model = AutoModel.from_pretrained(AUDIO_MODEL_CARD).to(device)

        self.dataset = []

        # Iterate over df
        for _, row in tqdm(df.iterrows()):
            path = os.path.join(self.audio_dir, f"{row['Document']}/{row['idClip']}.wav")
            if os.path.exists(path):
                # obtain audio WAV2VEC features
                audio, sampling_rate = torchaudio.load(path)
                # resample audio if necessary
                if sampling_rate != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
                    # mean pooling over channels
                    audio = torch.mean(audio, dim=0, keepdim=True)
                with torch.inference_mode():
                    # run audio through model
                    input_values = self.audio_processor(audio, sampling_rate=self.sample_rate).input_values[0]
                    input_values = torch.tensor(input_values).to(device)
                    audio_model_output = self.audio_model(input_values)
                    audio_features = audio_model_output.last_hidden_state[0].unsqueeze(0)
                    
                    # TODO: this step should not be necessary with more computational power
                    # downsample audio features
                    audio_features = torch.nn.functional.interpolate(audio_features.permute(0,2,1), scale_factor=DOWNSAMPLE_FACTOR, mode='linear')
                    audio_features = audio_features.permute(0,2,1)[0]

                    audio_features = audio_features.cpu()
                
                text = row['Text']

                self.dataset.append((text, audio_features, LABEL_2_ID[row['Component']]))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def load_datasets(train_df, val_df, test_df):
    """
    Load the datasets from memory if possible, otherwise create them and save them to memory
    Args:
        train_df: training dataframe
        val_df: validation dataframe
        test_df: test dataframe
    """
    try:
        # try to restore datasets from memory
        train_dataset = torch.load(f'{LOAD_DATASET_PATH}/train_dataset.pkl')
        val_dataset = torch.load(f'{LOAD_DATASET_PATH}/val_dataset.pkl')
        test_dataset = torch.load(f'{LOAD_DATASET_PATH}/test_dataset.pkl')
    except:
        train_dataset = MM_Dataset(train_df, AUDIO_PATH, SAMPLE_RATE)
        val_dataset = MM_Dataset(val_df, AUDIO_PATH, SAMPLE_RATE)
        test_dataset = MM_Dataset(test_df, AUDIO_PATH, SAMPLE_RATE)

        torch.save(train_dataset, f'{LOAD_DATASET_PATH}/train_dataset.pkl')
        torch.save(val_dataset, f'{LOAD_DATASET_PATH}/val_dataset.pkl')
        torch.save(test_dataset, f'{LOAD_DATASET_PATH}/test_dataset.pkl')
    
    if REMOVE_OTHER:
        train_dataset = list(filter(lambda x: x[2] != OTHER_LABEL_ID, train_dataset))
        val_dataset = list(filter(lambda x: x[2] != OTHER_LABEL_ID, val_dataset))
        test_dataset = list(filter(lambda x: x[2] != OTHER_LABEL_ID, test_dataset))
    return train_dataset, val_dataset, test_dataset


def create_split_dataloader(dataset, batch_size):
    """
    Create a DataLoader object from the given dataset with the given batch size
    Args:
        dataset: dataset to use
        batch_size: batch size to use
    """
    def pack_fn(batch):
        """
        Function to pad the audio features and create the attention mask
        """
        texts = [x[0] for x in batch]
        audio_features = [x[1] for x in batch]
        labels = torch.tensor([x[2] for x in batch])
        
        # pad audio features
        audio_features = pad_sequence(audio_features, batch_first=True, padding_value=float('-inf'))
        audio_features_attention_mask = audio_features[:, :, 0] != float('-inf')
        audio_features[(audio_features == float('-inf'))] = 0
        return texts, audio_features, audio_features_attention_mask, labels

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pack_fn)
    return dataloader


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """
    Create the dataloaders for the train, validation and test sets
    Args:
        train_dataset: training dataset
        val_dataset: validation dataset
        test_dataset: test dataset
    """
    train_dataloader = create_split_dataloader(train_dataset, BATCH_SIZE)
    val_dataloader = create_split_dataloader(val_dataset, BATCH_SIZE)
    test_dataloader = create_split_dataloader(test_dataset, BATCH_SIZE)
    return train_dataloader, val_dataloader, test_dataloader