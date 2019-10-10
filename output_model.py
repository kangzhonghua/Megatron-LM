from pretrain_gpt2 import initialize_distributed
from utils import load_checkpoint
from model import GPT2Model
import os
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_dir = "/home/Public/Megatron-LM/checkpoints/gpt2_87.75m_hm8g"
hf_model_dir = os.path.join(model_dir, "hf")

if not os.path.exists(hf_model_dir):
    os.makedirs(hf_model_dir)


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

    print('building GPT2 model ...\n')
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

    return model


args = c_args()

initialize_distributed(args)

model = get_model()

_ = load_checkpoint(
    model, None, None, args)


config_class = GPT2Config(
        vocab_size_or_config_json_file=32128,
        n_positions=768,
        n_ctx=768,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,

        num_labels=1,
        summary_type='cls_index',
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
)

print(config_class)

config_class.save_pretrained(hf_model_dir)

from utils import *
gpt2model = GPT2LMHeadModel(config_class)

print(gpt2model)

move_weights(model,gpt2model, dst2src=True)

gpt2model.save_pretrained(hf_model_dir)

print("end.")
