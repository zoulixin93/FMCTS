#!/usr/bin/env bash

m100k(){
    ~/pyenv3/bin/python3 ./train_file.py -model $1 -log_path ./log/ -data_name m100k \
    -root_path ./data/ml-100k/ -rating_path rat.dat -cat_path cat.dat -item_num 1683 \
    -user_num 944 -cat_n1_num 7 -cat_n2_num 20 -c_puct $2 -n_playout $3 -temperature $4 \
    -update_frequency $5 -batch_size $6 -epoch $7 -memory_capacity $8 -learning_rate $9 \
    -optimizer_name ${10} -evaluate_num ${11} -latent_factor ${12} -delete_previous ${13} \
    -job_ports ${14} -task ${15} -evaluate_num 500
}

name="m100k_hmcts"
(m100k $name 20.0 50 20.0 1 60 10000 1000 0.005 sgd 100 20 False [40001] train) &
(m100k $name 20.0 50 20.0 1 60 10000 1000 0.005 sgd 100 20 False [40001] evaluate) &



