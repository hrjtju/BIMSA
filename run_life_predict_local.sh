# wandb setup 
export WANDB_BASE_URL=https://api.bandw.top
wandb online

# Check if any arguments are passed
if [ $# -gt 0 ]
then
    # If arguments are passed, use the first argument as the script path
    BIMSA_LIFE_100_DIR="./predictor_life/datasets/life/" python ./predictor_life/train_test.py -p $0
else
    # If no arguments are passed, use the default path
    BIMSA_LIFE_100_DIR="./predictor_life/datasets/life/" python ./predictor_life/train_test.py
fi