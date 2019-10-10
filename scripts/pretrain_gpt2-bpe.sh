#! /bin/bash

# Runs the "110M" parameter model
RANK=0
WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=1

python pretrain_gpt2.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --batch-size 20 \
    --seq-length 768 \
    --max-position-embeddings 768 \
    --train-iters 320000 \
    --save checkpoints/gpt2_110m \
    --load checkpoints/gpt2_110m \
    --resume-dataloader \
    --train-data /home/Public/Megatron-LM/data/wikipedia/wikipedia.json \
    --loose-json \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path /home/Public/data/gpt2/output/gpt2_huamei_corpus.bpe_src.small/gpt2_huamei_corpus_bpe_32k_v2.model \
    --cache-dir cache  --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0  \
    --warmup .01 \
    --checkpoint-activations \
    --lazy-loader \
    --vocab-size 32000 \
    --text-key text \
    --fp16 \
    --save-interval 1000 

set +x
