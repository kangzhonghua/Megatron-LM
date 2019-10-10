import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0


class c_args(object):
    def __init__(self):
        self.distributed_backend = "nccl"
        self.world_size = 1
        self.rank = 0
        self.load = "/home/Public/Megatron-LM/checkpoints/gpt2_87.75m_hm8g"
        self.finetune = False
        self.no_load_optim = True
        self.no_load_rng = True
        self.model_parallel_size = 1
        self.local_rank = 0
        self.fp16 = True


def get_model():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_layers=12,
                      vocab_size=32128,
                      hidden_size=768,
                      num_attention_heads=12,
                      embedding_dropout_prob=0.1,
                      attention_dropout_prob=0.1,
                      output_dropout_prob=0.1,
                      max_sequence_length=768,
                      checkpoint_activations=True,
                      checkpoint_num_layers=1,
                      parallel_output=False)

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
    model = DDP(model)

    return model


args = c_args()

initialize_distributed(args)

model = get_model()

_ = load_checkpoint(
    model, None, None, args)

print(model)