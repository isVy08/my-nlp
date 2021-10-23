
import torch, sys
from main import load_model
from transformer import Seq2SeqTransformer
from data_generator import to_tokenizer
from utils import get_config
from trainer import generate_square_subsequent_mask

def greedy_decoder(model, src, src_mask, max_len, BOS_IDX=2, EOS_IDX=3, DEVICE='cpu'):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        # out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys

def decode(sentence, model, tokenizer, tokenizer_type, DEVICE):
    
    if tokenizer_type == 'char':
        src = [2] + tokenizer.encode(sentence).ids + [3] 
    else:
        src = [2] + tokenizer.encode(sentence) + [3]
    src = torch.tensor(src).view(1,-1).to(DEVICE)
    num_tokens = src.shape[1]
    src_mask = torch.zeros((num_tokens, num_tokens)).type(torch.bool).to(DEVICE)
    tgt_tokens = greedy_decoder(model, src, src_mask, max_len=src.shape[1] + 5, DEVICE=DEVICE)
    return tokenizer.decode(tgt_tokens.tolist()[0])

if __name__ == "__main__":
    config_file, sentence = sys.arg[1], sys.arg[2]
    config = get_config()
    
    # Load tokenizer
    tokenizer = to_tokenizer(config.tokenizer_type, config.tokenizer_path)
    SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = tokenizer.get_vocab_size()
    
    # Load model
    model = Seq2SeqTransformer(config.NUM_ENCODER_LAYERS, config.NUM_DECODER_LAYERS, config.EMB_SIZE,
                                config.NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, config.FFN_HID_DIM, config.DROPOUT)

    load_model(model, optimizer=None, model_path=config.model_path, device=config.DEVICE)
    model.eval()
    prediction = decode(sentence, model, tokenizer, config.tokenizer_type, config.DEVICE)
    print(prediction)

