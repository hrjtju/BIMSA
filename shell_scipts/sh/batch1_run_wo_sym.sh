set BIMSA_LIFE_DIR=./predictor_life_simple/datasets
network_ls=('small_2_layer_seq_p4cnn' 'tiny_2_layer_seq_p4cnn' 'small_2_layer_seq_cnn' 'tiny_2_layer_seq_cnn' 'multiscale_0' 'multiscale_p4')
for network in ${network_ls[@]}; do
    for rule in "B3/S23" "B36/S23" "B3678/S34678" "B35678/S5678"; do 
        echo rule: $rule "\t", network: $network "\n\n"
        python ./predictor_life_simple/train_test.py -r $rule -p "./predictor_life_simple/hyperparams/$network.toml"
    done
done

# python ./apredictor_life_simple/train_test.py -r "B3/S23" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B36/S23" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B3678/S34678" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B35678/S5678" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B2/S" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B345/S5" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B13/S012V" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B2/S013V" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_p4cnn.toml"

# python ./predictor_life_simple/train_test.py -r "B3/S23" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B36/S23" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B3678/S34678" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B35678/S5678" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B2/S" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B345/S5" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B13/S012V" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"
# python ./predictor_life_simple/train_test.py -r "B2/S013V" -p "./predictor_life_simple/hyperparams/small_2_layer_seq_cnn.toml"

# python ./predictor_life_simple/train_test.py -r "B3/S23" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"
# python ./predictor_life_simple/train_test.py -r "B36/S23" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"
# python ./predictor_life_simple/train_test.py -r "B3678/S34678" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"
# python ./predictor_life_simple/train_test.py -r "B35678/S5678" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"
# python ./predictor_life_simple/train_test.py -r "B2/S" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"
# python ./predictor_life_simple/train_test.py -r "B345/S5" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"
# python ./predictor_life_simple/train_test.py -r "B13/S012V" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"
# python ./predictor_life_simple/train_test.py -r "B2/S013V" -p "./predictor_life_simple/hyperparams/multiscale_0.toml"

