rem wandb setup 

set WANDB_BASE_URL=https://api.bandw.top
wandb online

@echo off

rem Check if any arguments are passed
if "%~1"=="" (
    rem If no arguments are passed, use the default path
    set "BIMSA_LIFE_100_DIR=.\predictor_life\datasets\life\"
    python .\predictor_life_simple\train_test.py
) else (
    rem If arguments are passed, use the first argument as the script path
    set "BIMSA_LIFE_100_DIR=.\predictor_life\datasets\life\"
    python .\predictor_life_simple\train_test.py -p "%~1"
)