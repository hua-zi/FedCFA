#!/bin/bash

CUDA_VISIBLE_DEVICES='3' python main.py \
    --topk 24 \
    --fedcfa_rate 1:5:5 \
    --alg fedcfa \
    --com_round 500 \
    --total_client 60 \
    --alpha 0.6 \
    --data_name CIFAR10 \
    --partition dirichlet