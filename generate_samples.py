# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from arguments import get_args
from utils import Timers
from utils import load_checkpoint_model
from data_utils.tokenization_gpt2 import GPT2Tokenizer
from configure_data import configure_data
import mpu

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0

from flask import Flask, request, abort

USE_TORCH_DDP = False

def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)

    return tokens, attention_mask, position_ids

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits


def generate_samples(raw_text, stop_words, model, tokenizer, args, device):
    model.eval()
    with torch.no_grad():
        torch.distributed.barrier(group=mpu.get_model_parallel_group())

        if mpu.get_model_parallel_rank() == 0:
            context_tokens = tokenizer.encode(raw_text)
            context_length = len(context_tokens)
        else:
            context_tokens = tokenizer.encode("空文本")
            context_length = len(context_tokens)
        
        pad_id = tokenizer.encoder['<pad>']
        args.eod_token = tokenizer.encoder['<eod>']
        stop_words = [tokenizer.encode(word) for word in stop_words]
        stop_words.append(args.eod_token)
        if context_length < args.seq_length:
            context_tokens.extend([pad_id] * (args.seq_length - context_length))

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor([context_length])

        torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

        context_length = context_length_tensor[0].item()
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

        start_time = time.time()

        counter = 0
        past_key_values = None
        while counter <  args.out_seq_length:
            if counter == 0:
                logits, past_key_values = model(
                    tokens[:, :context_length], 
                    position_ids[:, :context_length], 
                    attention_mask[:, :, :context_length, :context_length], 
                    past_key_values=past_key_values, 
                    use_cache=True)
                logits = logits[:, context_length - 1, :]
            else:
                logits, past_key_values = model(
                    tokens[:, context_length - 1 : context_length], 
                    position_ids[:, context_length - 1 : context_length], 
                    attention_mask[:, :, context_length - 1, :context_length], 
                    past_key_values=past_key_values, 
                    use_cache=True)
                logits = logits[:, 0, :]

            if args.fp16: 
                past_key_values = [x.half() for x in past_key_values]
            else:
                past_key_values = [x for x in past_key_values]
            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)            
            log_probs = F.softmax(logits/args.temperature, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)

            sampled_token = prev[0]
            tokens[0, context_length] = sampled_token
            torch.distributed.broadcast(tokens, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            context_length += 1
            counter += 1

            print('{} in {}'.format(sampled_token, stop_words))
            if sampled_token in stop_words:
                break
        
        output_tokens_list = tokens.view(-1).contiguous()
        decode_tokens = tokenizer.decode(output_tokens_list.tolist())
        trim_decode_tokens = decode_tokens[len(raw_text):decode_tokens.find("<eod>")]

        torch.distributed.barrier(group=mpu.get_model_parallel_group())
        return trim_decode_tokens


def prepare_tokenizer(args):

    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    args.tokenizer_num_tokens = tokenizer.num_tokens
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    after = tokenizer.num_tokens
    while after % mpu.get_model_parallel_world_size() != 0:
        after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def get_model(args):
    """Build the model."""

    print_rank_0('building CPM model ...')
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=args.parallel_output)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model


def setup_model(args):
    """Setup model."""

    model = get_model(args)

    args.iteration = load_checkpoint_model(model, args)

    return model


def main():

    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = GPT2Tokenizer(
        os.path.join(args.tokenizer_path, 'vocab.json'), 
        os.path.join(args.tokenizer_path, 'chinese_vocab.model')
    )

    # Model
    args.parallel_output = False
    model = setup_model(args)

    #setting default batch size to 1
    args.batch_size = 1

    PROMPT_KEY = 'prompt'
    MAX_TOKENS_KEY = 'max_length'
    TEMPERATURE_KEY = 'temperature'
    TOP_K_KEY = 'top_k'
    TOP_P_KEY = 'top_p'
    STOP_WORDS_KEY = 'stop_words'

    DEFAULT_MAX_TOKEN = 128
    DEFAULT_TEMPERATURE = 0.85
    DEFAULT_TOP_P = 0.8
    DEFAULT_TOP_K = 50

    app = Flask(__name__)

    @app.route('/generation', methods=['POST'])
    def generation():
        if not request.json or not PROMPT_KEY in request.json or not STOP_WORDS_KEY in request.json:
            abort(400)

        prompt = request.json[PROMPT_KEY]
        stop_words = request.json[STOP_WORDS_KEY]
        max_token = request.json[MAX_TOKENS_KEY] if MAX_TOKENS_KEY in request.json else DEFAULT_MAX_TOKEN
        temperature = request.json[TEMPERATURE_KEY] if TEMPERATURE_KEY in request.json else DEFAULT_TEMPERATURE
        top_p = request.json[TOP_P_KEY] if TOP_P_KEY in request.json else DEFAULT_TOP_P
        top_k = request.json[TOP_K_KEY] if TOP_K_KEY in request.json else DEFAULT_TOP_K

        args.out_seq_length = max_token
        args.temperature = temperature
        args.top_k = top_k
        args.top_p = top_p
        result = generate_samples(prompt, stop_words, model, tokenizer, args, torch.cuda.current_device())

        return result, 200

    app.run()


if __name__ == '__main__':
    main()
