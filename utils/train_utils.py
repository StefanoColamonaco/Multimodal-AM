import torch
import copy
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pickle


from utils.models import *
from utils.CustomTransformer import *
from utils.constants import *
from utils.data_utils import *


class BestModel:
    """
    Class to keep track of the best performing model on validation set during training
    """
    def __init__(self):
        self.best_validation_loss = float('Infinity')
        self.best_state_dict = None
    def __call__(self, model, loss):
        if loss < self.best_validation_loss:
            self.best_validation_loss = loss
            self.best_state_dict = copy.deepcopy(model.state_dict())


def evaluate(model, data_loader, loss_fn, debug=False):
    """
    Evaluate the model on the set passed
    Args:
        model: model to evaluate
        data_loader: DataLoader object
        loss_fn: loss function to use
        debug: whether to print debug statements
    """
    model.eval()
    total_loss = 0.0
    num_correct = 0 
    num_examples = 0
    tot_pred, tot_targ, tot_logits = torch.LongTensor(), torch.LongTensor(), torch.LongTensor()
    for batch in data_loader:
        texts, audio_features, audio_attention, targets = batch
        audio_features = audio_features.to(device)
        audio_attention = audio_attention.to(device)
        targets = targets.to(device)
        output = model(texts,audio_features,audio_attention)
        if debug:
            print("OUTPUT",output)
            print("TARGETS", targets)
        loss = loss_fn(output, targets)
        total_loss += loss.detach()
        
        # if label O is still in the dataset we remove it from the outputs
        # since it's a binary task
        if not REMOVE_OTHER:
            not_other = targets != 2
            output = output[not_other]
            targets = targets[not_other]
        
        scores = output[:, :2]
        predicted_labels = torch.argmax(scores, dim=-1)

        tot_pred = torch.cat((tot_pred, predicted_labels.detach().cpu()))
        tot_targ = torch.cat((tot_targ, targets.detach().cpu()))
        tot_logits = torch.cat((tot_logits, torch.nn.functional.softmax(scores, dim=-1)[:, 1].detach().cpu()))       

        correct = torch.eq(predicted_labels, targets).view(-1)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
    total_loss = total_loss.cpu().item()
    total_loss /= len(data_loader.dataset)
    accuracy = num_correct/num_examples
    f1 = multiclass_f1_score(tot_pred, tot_targ, num_classes=2, average="macro")
    return total_loss, accuracy, f1, tot_pred, tot_targ, tot_logits

            
def train(model, loss_fn, train_loader, val_loader, epochs=10, device="cuda", lr=1e-3, lr_decay_factor=0.1, lr_decay_patience=3, weight_decay=1e-5, verbose=True, debug=False):
    """
    Train the model on the train set and evaluate on the validation set with the given parameters
    Args:
        model: model to train
        loss_fn: loss function to use
        train_loader: DataLoader object for train set
        val_loader: DataLoader object for validation set
        epochs: number of epochs
        device: device to use
        lr: initial learning rate
        lr_decay_factor: factor to decay learning rate
        lr_decay_patience: patience for learning rate decay
        weight_decay: weight decay
        verbose: whether to print training results
        debug: whether to print debug statements
    """
    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=lr_decay_patience, verbose=True)
    best_model_tracker = BestModel()
    # history of train and validation losses, accuracy and f1
    history_train_losses = []
    history_train_accuracy = []
    history_train_f1 = []

    history_val_losses = []
    history_val_accuracy = []
    history_val_f1 = []

    for epoch in tqdm(range(epochs)):
        # training
        correct = 0
        training_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            texts, audio_features, audio_attention, targets = batch
            audio_features = audio_features.to(device)
            audio_attention = audio_attention.to(device)
            targets = targets.to(device)
            output = model(texts,audio_features,audio_attention)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            correct += torch.eq(torch.argmax(output, dim=-1), targets).view(-1).sum()
            training_loss += loss.detach()
        training_loss = training_loss.cpu().item()

        training_loss /= len(train_loader.dataset)
        training_accuracy = correct.item() / len(train_loader.dataset)
        training_f1 = multiclass_f1_score(torch.argmax(output, dim=-1), targets, num_classes=2, average="macro")

        valid_loss, valid_accuracy, valid_f1, _, _, _ = evaluate(model, val_loader, loss_fn, debug)

        history_train_losses.append(training_loss)
        history_train_accuracy.append(training_accuracy)
        history_train_f1.append(training_f1)

        history_val_losses.append(valid_loss)
        history_val_accuracy.append(valid_accuracy)
        history_val_f1.append(valid_f1)

        best_model_tracker(model, valid_loss)
        scheduler.step(valid_loss)
        if verbose:
            print(f'Epoch: {epoch}, Training Loss: {training_loss:.4f}, Validation Loss: {valid_loss:.4f}, accuracy = {valid_accuracy:.4f}, F1={valid_f1:.4f}')
    # restore best model weights
    model.load_state_dict(best_model_tracker.best_state_dict) 
    history = {
        'train_loss': history_train_losses,
        'train_accuracy': history_train_accuracy,
        'train_f1': history_train_f1,
        'val_loss': history_val_losses,
        'val_accuracy': history_val_accuracy,
        'val_f1': history_val_f1
    }
    return model, history


def create_models(
        tokenizer, embedder,
        head_hidden_dimension=HEAD_HIDDEN_DIMENSION, dropout_prob=DROPOUT_PROB, hidden_state_index=HIDDEN_STATE_INDEX,   # shared parameters
        audio_only_nheads=AUDIO_ONLY_N_HEADS, audio_only_d_ffn=AUDIO_ONLY_D_FFN, audio_only_n_layers=AUDIO_ONLY_N_LAYERS, # audio only parameters
        csa_n_heads=CSA_N_HEADS, csa_d_ffn=CSA_D_FFN, csa_n_layers=CSA_N_LAYERS, # multimodal parameters
        ensembling_n_heads=ENSEMBLING_N_HEADS, ensembling_d_ffn=ENSEMBLING_D_FFN, ensembling_n_layers=ENSEMBLING_N_LAYERS, # ensembling parameters
        multa_nblocks=MULTA_N_BLOCKS, multa_d_ffn=MULTA_D_FFN # unaligned parameters
    ):
    """
    Helper function to create and return all the models
    """
    ###################################################################################### -- TEXT MODEL --
    text_only_head = nn.Sequential(
        nn.Linear(EMBEDDING_DIM, head_hidden_dimension),
        nn.ReLU(),
        nn.Linear(head_hidden_dimension, MODEL_NUM_LABELS)
    ).to(device)
    text_only = TextModel(tokenizer, embedder, text_only_head)


    ###################################################################################### -- AUDIO MODEL --   
    audio_only_head = nn.Sequential(
        nn.Linear(EMBEDDING_DIM, head_hidden_dimension),
        nn.ReLU(),
        nn.Linear(head_hidden_dimension, MODEL_NUM_LABELS)
    ).to(device)
    audio_only_transformer_layer = nn.TransformerEncoderLayer(d_model=EMBEDDING_DIM, nhead=audio_only_nheads, dim_feedforward=audio_only_d_ffn, batch_first=True).to(device)
    audio_only_transformer_encoder = nn.TransformerEncoder(audio_only_transformer_layer, num_layers=audio_only_n_layers).to(device)
    audio_only = AudioModel(audio_only_transformer_encoder, audio_only_head).to(device)


    ###################################################################################### -- MULTIMODAL MODEL --
    multimodal_encoder = CustomEncoder(d_model=EMBEDDING_DIM, ffn_hidden=csa_d_ffn, n_head=csa_n_heads, n_layers=csa_n_layers, drop_prob=dropout_prob)
    multimodal_transformer_head = nn.Sequential(
        nn.Linear(EMBEDDING_DIM, head_hidden_dimension),
        nn.ReLU(),
        nn.Linear(head_hidden_dimension, MODEL_NUM_LABELS)
    ).to(device)
    multimodal_transformer = CSA(tokenizer, embedder, multimodal_encoder, multimodal_transformer_head, hidden_state_index=hidden_state_index).to(device)


    ###################################################################################### -- ENSEMBLING MODEL --
    ensembling_text_head = nn.Sequential(
        nn.Linear(EMBEDDING_DIM, head_hidden_dimension),
        nn.ReLU(),
        nn.Linear(head_hidden_dimension, MODEL_NUM_LABELS)
    ).to(device)
    ensembling_audio_head = nn.Sequential(
        nn.Linear(EMBEDDING_DIM, head_hidden_dimension),
        nn.ReLU(),
        nn.Linear(head_hidden_dimension, MODEL_NUM_LABELS)
    ).to(device)
    ensembling_transformer_layer = nn.TransformerEncoderLayer(d_model=EMBEDDING_DIM, nhead=ensembling_n_heads, dim_feedforward=ensembling_d_ffn, batch_first=True).to(device)
    ensembling_transformer_encoder = nn.TransformerEncoder(ensembling_transformer_layer, num_layers=ensembling_n_layers).to(device)
    ensembling_text_model = TextModel(tokenizer, embedder, ensembling_text_head)
    ensembling_audio_model = AudioModel(ensembling_transformer_encoder, ensembling_audio_head)
    ensembling_fusion = Ensembling(ensembling_text_model, ensembling_audio_model).to(device)


    ###################################################################################### -- UNALIGNED MODEL --
    unaligned_head = nn.Sequential(
        nn.Linear(EMBEDDING_DIM*2, head_hidden_dimension),
        nn.ReLU(),
        nn.Linear(head_hidden_dimension, MODEL_NUM_LABELS)
    ).to(device)
    unaligned_mm_model = MulTA(embedding_dim=EMBEDDING_DIM, d_ffn=multa_d_ffn, n_blocks=multa_nblocks, head=unaligned_head, hidden_state_index=hidden_state_index, dropout_prob=dropout_prob).to(device)
    

    ###################################################################################### -- RETURN --
    return [text_only, audio_only, multimodal_transformer, ensembling_fusion, unaligned_mm_model]    


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train_evaluate_all_models(class_weight, train_dataloader, val_dataloader, test_dataloader, tokenizer, embedder):
    # results
    val_results = {n : [] for n in MODEL_NAMES}
    test_results = {n : [] for n in MODEL_NAMES}
    
    # history
    history_train_losses = {n : [] for n in MODEL_NAMES}
    history_train_accuracy = {n : [] for n in MODEL_NAMES}
    history_train_f1 = {n : [] for n in MODEL_NAMES}
    history_val_losses = {n : [] for n in MODEL_NAMES}
    history_val_accuracy = {n : [] for n in MODEL_NAMES}
    history_val_f1 = {n : [] for n in MODEL_NAMES}

    for seed in SEEDS:
        print(f'{f"TRAINING WITH SEED {seed}":=^100}')
        print()
        set_seed(seed)
        models = create_models(tokenizer, embedder)
        while models:
            model = models[0]
            model_name = MODEL_NAMES[0]
            set_seed(seed)
            print(f'{f"Training model {model_name}":_^100}')

            loss = nn.CrossEntropyLoss(weight=class_weight)

            if model_name == 'Ensembling':
                loss = lambda outputs, targets: torch.nn.functional.nll_loss(torch.log(outputs), targets, weight=weight, reduction='mean')
            model, history = train(
                model,
                loss,
                train_dataloader,
                val_dataloader,
                epochs=EPOCHS,
                device=device,
                lr=INITIAL_LR,
                lr_decay_factor=LR_DECAY_FACTOR,
                lr_decay_patience=LR_DECAY_PATIENCE,
                weight_decay=WEIGHT_DECAY,
                verbose=VERBOSE_TRAIN,
                debug = DEBUG_TRAIN
            )

            # save history
            history_train_losses[model_name].append(history['train_loss'])
            history_train_accuracy[model_name].append(history['train_accuracy'])
            history_train_f1[model_name].append(history['train_f1'])
            history_val_losses[model_name].append(history['val_loss'])
            history_val_accuracy[model_name].append(history['val_accuracy'])
            history_val_f1[model_name].append(history['val_f1'])

            # evaluate on validation set
            _, val_acc, val_f1, val_pred, val_targ, val_logits = evaluate(model, val_dataloader, loss)
            val_results[model_name].append({'acc': val_acc, 'f1': val_f1, 'pred': val_pred, 'targ': val_targ, 'logits':val_logits})

            # evaluate on test set
            _, test_acc, test_f1, test_pred, test_targ, test_logits = evaluate(model, test_dataloader, loss)
            test_results[model_name].append({'acc': test_acc, 'f1': test_f1, 'pred': test_pred, 'targ': test_targ, 'logits':test_logits})

            if VERBOSE_TRAIN:
                print(f'[VAL] Model: {model_name} - acc: {val_acc:.4f} - f1: {val_f1:.4f}')
                print(f'[TEST] Model: {model_name} - acc: {test_acc:.4f} - f1: {test_f1:.4f}')
                print()

            # save weights
            torch.save(model.state_dict(), f'{WEIGHTS_PATH}/{model_name}_seed_{seed}.pt')

            del model
            del models[0]
            del MODEL_NAMES[0]
            gc.collect()
    # save train history
    with open(f'{HISTORY_PATH}/train_losses.pkl', 'wb') as f:
        pickle.dump(history_train_losses, f)
    with open(f'{HISTORY_PATH}/train_accuracy.pkl', 'wb') as f:
        pickle.dump(history_train_accuracy, f)
    with open(f'{HISTORY_PATH}/train_f1.pkl', 'wb') as f:
        pickle.dump(history_train_f1, f)
    
    # save val history
    with open(f'{HISTORY_PATH}/val_losses.pkl', 'wb') as f:
        pickle.dump(history_val_losses, f)
    with open(f'{HISTORY_PATH}/val_accuracy.pkl', 'wb') as f:
        pickle.dump(history_val_accuracy, f)
    with open(f'{HISTORY_PATH}/val_f1.pkl', 'wb') as f:
        pickle.dump(history_val_f1, f)

    # save val results
    with open(f'{RESULTS_PATH}/val_results.pkl', 'wb') as f:
        pickle.dump(val_results, f)

    # save test results
    with open(f'{RESULTS_PATH}/test_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)

