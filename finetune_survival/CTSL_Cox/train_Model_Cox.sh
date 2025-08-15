#!/bin/bash

trap "echo 'Ctrl+C detected. Killing all child processes...'; pkill -P $$; exit 1" SIGINT

train_file_root="finetune_survival/CTSL_Cox/train_CTSL_Cox_IECV.py"
model_id="uniformer_distill"
python $train_file_root model.name=$model_id data.dataset="renji" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="lasso" &
python $train_file_root model.name=$model_id data.dataset="renji" tricks.apply_strata=False tricks.apply_stepwise=False &
python $train_file_root model.name=$model_id data.dataset="renji" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="stepwise" &

wait

python $train_file_root model.name=$model_id data.dataset="anzhen" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="lasso" &
python $train_file_root model.name=$model_id data.dataset="anzhen" tricks.apply_strata=False tricks.apply_stepwise=False &
python $train_file_root model.name=$model_id data.dataset="anzhen" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="stepwise" &

wait

python $train_file_root model.name=$model_id data.dataset="tongji" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="lasso" &
python $train_file_root model.name=$model_id data.dataset="tongji" tricks.apply_strata=False tricks.apply_stepwise=False &
python $train_file_root model.name=$model_id data.dataset="tongji" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="stepwise" &

wait

python $train_file_root model.name=$model_id data.dataset="gulou" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="lasso" &
python $train_file_root model.name=$model_id data.dataset="gulou" tricks.apply_strata=False tricks.apply_stepwise=False &
python $train_file_root model.name=$model_id data.dataset="gulou" tricks.apply_strata=False tricks.apply_stepwise=True tricks.feature_selection_method="stepwise" &
