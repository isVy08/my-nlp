import torch.nn as nn
import sys, torch, os
from utils import Namespace, get_config
from timeit import default_timer as timer
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from transformer import PositionalEncoding, TokenEmbedding, Seq2SeqTransformer
from trainer import create_mask, generate_square_subsequent_mask, train_epoch, evaluate


torch.manual_seed(8)

# Data loader

def load_model(model, optimizer, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def run(config):

    DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    train_source, train_target, val_source, val_target = config.dataset_path
    path_to_train_input, path_to_val_input = config.input_path

    # Load data
    train_iter = DataGenerator(path_to_train_input, train_source, train_target, 'train', 
                                                config.tokenizer_path, config.tokenizer_type,
                                                config.vocab_size, config.min_frequency, config.data_fraction)
    
    val_iter = DataGenerator(path_to_val_input, val_source, val_target, 'val', 
                                            config.tokenizer_path, config.tokenizer_type,
                                            config.vocab_size, config.min_frequency)

    train_loader = DataLoader(train_iter, batch_size=config.batch_size)
    val_loader = DataLoader(val_iter, batch_size=config.batch_size)

    # Build model
    SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = train_iter.vocab_size
    model = Seq2SeqTransformer(config.NUM_ENCODER_LAYERS, config.NUM_DECODER_LAYERS, config.EMB_SIZE,
                                    config.NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, config.FFN_HID_DIM, config.DROPOUT)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=config.eps)

    if os.path.isfile(config.model_path):
        load_model(model, optimizer, config.model_path, DEVICE)
    else:
        model.to(DEVICE)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_iter.pad_idx)

    # Training
    print('Training begins ...')
    NUM_EPOCHS = config.epochs
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, train_loader, loss_fn, DEVICE)
       
        # Checkpoint
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                     }, config.model_path)
        
        # Evaluation
        val_loss = evaluate(model, val_loader, loss_fn, DEVICE)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"
        print(msg)

if __name__ == "__main__":
    config_file = sys.argv[1]
    config = get_config(config_file)
    run(config)