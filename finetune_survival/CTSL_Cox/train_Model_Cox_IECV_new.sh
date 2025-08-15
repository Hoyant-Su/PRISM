#!/bin/bash

trap "echo 'Ctrl+C detected. Killing all child processes...'; pkill -P $$; exit 1" SIGINT
model_dir="ssl_pretraining"
root="./"
train_file_root="$root/finetune_survival/CTSL_Cox/train_CTSL_Cox_IECV.py"
model_id="PRISM"
exp_str="0515_attention_viz_prompt_200/projector/2025-05-21 06:15:50"

ehr_group="all"

filename_with_ext="${dl_csv
inference_prompt="${filename_with_ext%.csv}"
inference_prompt="${inference_prompt

run_internal_setting() {
    local dl_csv="$1"
    local inference_prompt="$2"
    local setting="internal"
    local datasets=("renji" "anzhen" "tongji" "gulou")
    local feature_methods=("lasso" "stepwise")
    local seed_list=("256" "634" "9323" "55793" "57131")

    for seed in "${seed_list[@]}"; do
        echo "🔁 Running seed: $seed"
        for dataset in "${datasets[@]}"; do
            for method in "${feature_methods[@]}"; do
                python "$train_file_root" model.name="$model_id" \
                    data.dataset="$dataset" \
                    tricks.apply_strata=False \
                    tricks.apply_stepwise=True \
                    tricks.feature_selection_method="$method" \
                    data.inference_prompt="$inference_prompt" \
                    data.dl_csv="$dl_csv" \
                    data.ehr_group="$ehr_group" \
                    setting="$setting" \
                    hyperparam.seed="$seed" &
            done

            python "$train_file_root" model.name="$model_id" \
                data.dataset="$dataset" \
                tricks.apply_strata=False \
                tricks.apply_stepwise=False \
                data.inference_prompt="$inference_prompt" \
                data.dl_csv="$dl_csv" \
                data.ehr_group="$ehr_group" \
                setting="$setting" \
                hyperparam.seed="$seed" &

        done
        wait
    done
}

run_external_setting() {
    local dl_csv="$1"
    local inference_prompt="$2"
    local setting="external"
    local datasets=("renji" "anzhen" "tongji" "gulou")
    local strata_opts=(False True)
    local stepwise_opts=(False True)
    local feature_methods=("lasso" "stepwise")

    local seed_list=("256" "634" "9323" "55793" "57131")

    for seed in "${seed_list[@]}"; do
        echo "🔁 Running seed: $seed"

        for dataset in "${datasets[@]}"; do
            for strata in "${strata_opts[@]}"; do
                for stepwise in "${stepwise_opts[@]}"; do
                    if [ "$stepwise" == "True" ]; then
                        for method in "${feature_methods[@]}"; do
                            python "$train_file_root" model.name="$model_id" \
                                data.dataset="$dataset" \
                                tricks.apply_strata="$strata" \
                                tricks.apply_stepwise="$stepwise" \
                                tricks.feature_selection_method="$method" \
                                data.inference_prompt="$inference_prompt" \
                                data.dl_csv="$dl_csv" \
                                data.ehr_group="$ehr_group" \
                                setting="$setting" \
                                hyperparam.seed="$seed" &
                        done
                    else
                        python "$train_file_root" model.name="$model_id" \
                            data.dataset="$dataset" \
                            tricks.apply_strata="$strata" \
                            tricks.apply_stepwise="$stepwise" \
                            data.inference_prompt="$inference_prompt" \
                            data.dl_csv="$dl_csv" \
                            data.ehr_group="$ehr_group" \
                            setting="$setting" \
                            hyperparam.seed="$seed" &
                    fi
                done
            done

        done
        wait
    done
}

dl_csv="${model_dir}/results/${exp_str}/output_features_Place <Physiological> and <Pharmaceutical> as your central criteria for integrati.csv"
ehr_group="all"

filename_with_ext="${dl_csv
inference_prompt="${filename_with_ext%.csv}"
inference_prompt="${inference_prompt

run_internal_setting "$dl_csv" "$inference_prompt"
wait

dl_csv="${model_dir}/results/${exp_str}/output_features_Place <Physiological> and <Pharmaceutical> as your central criteria for integrati.csv"
ehr_group="physiological_pharmaceutical"

filename_with_ext="${dl_csv
inference_prompt="${filename_with_ext%.csv}"
inference_prompt="${inference_prompt

run_internal_setting "$dl_csv" "$inference_prompt"
wait

