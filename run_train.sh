#!/bin/bash

MODEL_NAME="neus"

if [ "$MODEL_NAME" == "neus-facto" ]; then
    ns-train neus-facto \
        --pipeline.model.sdf-field.inside-outside False \
        --vis viewer \
        --experiment-name neus-facto-dtu24 sdfstudio-data \
        --data ../data/dtu/scan24/

elif [ "$MODEL_NAME" == "nerfacto" ]; then
    ns-train nerfacto \
        --vis viewer \
        --experiment-name nerfacto-dtu24 \
        sdfstudio-data --data ../data/dtu/scan24/ \

elif [ "$MODEL_NAME" == "instant-ngp" ]; then
    ns-train instant-ngp \
        --vis viewer \
        --experiment-name instant-ngp-dtu24 \
        sdfstudio-data --data ../data/dtu/scan24/ \

elif [ "$MODEL_NAME" == "unisurf" ]; then
    ns-train unisurf \
        --pipeline.model.sdf-field.use-grid-feature True \
        --pipeline.model.sdf-field.hidden-dim 256 \
        --pipeline.model.sdf-field.num-layers 2 \
        --pipeline.model.sdf-field.num-layers-color 2 \
        --pipeline.model.sdf-field.use-appearance-embedding False \
        --pipeline.model.sdf-field.geometric-init True \
        --pipeline.model.sdf-field.inside-outside False  \
        --pipeline.model.sdf-field.bias 0.5 \
        --trainer.steps-per-eval-image 5000 \
        --pipeline.datamanager.train-num-rays-per-batch 2048 \
        --pipeline.model.background-model none \
        --vis viewer \
        --experiment-name unisurf-dtu24 sdfstudio-data \
        --data ../data/dtu/scan24

elif [ "$MODEL_NAME" == "volsdf" ]; then
    ns-train volsdf \
        --pipeline.model.sdf-field.use-grid-feature True \
        --pipeline.model.sdf-field.hidden-dim 256 \
        --pipeline.model.sdf-field.num-layers 2 \
        --pipeline.model.sdf-field.num-layers-color 2 \
        --pipeline.model.sdf-field.use-appearance-embedding False \
        --pipeline.model.sdf-field.geometric-init True \
        --pipeline.model.sdf-field.inside-outside False  \
        --pipeline.model.sdf-field.bias 0.5 \
        --pipeline.model.sdf-field.beta-init 0.1 \
        --trainer.steps-per-eval-image 5000 \
        --pipeline.datamanager.train-num-rays-per-batch 2048 \
        --pipeline.model.background-model none \
        --vis viewer \
        --experiment-name volsdf-dtu24  sdfstudio-data \
        --data ../data/dtu/scan24

elif [ "$MODEL_NAME" == "neus" ]; then
    ns-train neus \
        --pipeline.model.sdf-field.use-grid-feature True \
        --pipeline.model.sdf-field.hidden-dim 256 \
        --pipeline.model.sdf-field.num-layers 2 \
        --pipeline.model.sdf-field.num-layers-color 2 \
        --pipeline.model.sdf-field.use-appearance-embedding False \
        --pipeline.model.sdf-field.geometric-init True \
        --pipeline.model.sdf-field.inside-outside False  \
        --pipeline.model.sdf-field.bias 0.5 \
        --pipeline.model.sdf-field.beta-init 0.3 \
        --trainer.steps-per-eval-image 5000 \
        --pipeline.datamanager.train-num-rays-per-batch 2048 \
        --pipeline.model.background-model none \
        --vis viewer \
        --experiment-name neus-dtu24  sdfstudio-data \
        --data ../data/dtu/scan24

fi