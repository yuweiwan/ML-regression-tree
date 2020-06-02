# ML-regression-tree
regression tree and gradient-boosted regression tree  
  
Command:  
  
Train a regression tree:  
python3 main.py --mode train --algorithm regression-tree --model-file regression.tree.model --train-data train.txt 
  
Run the trained model on development data:  
python3 main.py --mode test --model-file regression.tree.model --test-data dev.txt --predictions-file dev.predictions  
  
Use the script compute_mean_square_error.py to evaluate the root mean squared error of model's predictions:  
python3 compute_root_mean_square_error.py dev.txt dev.predictions
