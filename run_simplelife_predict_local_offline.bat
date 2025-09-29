rem wandb setup 

wandb offline

@echo off

rem Check if any arguments are passed
if "%~1"=="" (
    rem If no arguments are passed, use the default path
    set "BIMSA_LIFE_DIR=.\predictor_life\datasets\life\"
    python .\predictor_life_simple\train_test.py
) else (
    rem If arguments are passed, use the first argument as the script path
    set "BIMSA_LIFE_DIR=.\predictor_life\datasets\life\"
    python .\predictor_life_simple\train_test.py -p "%~1"
)