set BIMSA_LIFE_DIR=.\predictor_life_simple\datasets

python .\predictor_life_simple\train_test.py -r "B3/S23" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
python .\predictor_life_simple\train_test.py -r "B36/S23" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
python .\predictor_life_simple\train_test.py -r "B3678/S34678" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
python .\predictor_life_simple\train_test.py -r "B35678/S5678" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
python .\predictor_life_simple\train_test.py -r "B2/S" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
python .\predictor_life_simple\train_test.py -r "B345/S5" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
python .\predictor_life_simple\train_test.py -r "B13/S012V" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
python .\predictor_life_simple\train_test.py -r "B2/S013V" -p ".\predictor_life_simple\hyperparams\small_3_layer_seq_p4cnn.toml"
