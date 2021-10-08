#!/bin/bash

set -e

####
# Using different WDE models and baseline, generate and score translations for
# external domains
#
# Using 3 different WDE models and the baseline, we generate translations for
# domains outside the ones of BT. For each domain, a set of 100 samples is
# used to calculate the embedding and a set of 1000 samples from the corpus
# is translated. We redo this multiple times (bootstrap). The BLEU, chrF++
# and BERTScore are calculated.
####

models=(
  "AverageEncoderOutput;avg-enc-out/exp01"
  "DDE;dde/exp01"
  "SentenceBERT;sentence-bert/exp01"
)
base_out_dir=/part/01/Tmp/frenettx/couleurs/report-data/external-domains
evaluation_dir=$base_out_dir/evaluation
bpe_data_dir=$evaluation_dir/data/bpe-source
test_raw_data_dir=$base_out_dir/data/04-normalized
test_tokenized_data_dir=$base_out_dir/data/05-preprocessed
test_bpe_data_dir=$base_out_dir/data/06-bpe-bt
log_file=$base_out_dir/evaluate.log.txt
base_model_dir=/part/01/Tmp/frenettx/couleurs/baselines/trained_snmt/6jan2021_temperated/data
base_wde_dir=/part/01/Tmp/frenettx/couleurs/with-domain-token
nb_scorer_processes=5
seed=1984
model_config=/u/frenettx/maitrise/bt/experiments/with-domain-token/model-config.json
skip_to_bertscore=true

log()
{
  echo "$@" | tee -a $log_file
}

####
# Init
####

mkdir -p $base_out_dir $evaluation_dir
touch $log_file

if [[ $skip_to_bertscore == "true" ]]; then
  log "WARNING: skipping directly to calculating BERTScore"
else

  mkdir -p $bpe_data_dir

  ####
  # Create data dir by symlinking BPE source files and normalized (no BPE)
  # target files.
  ####

  log "Start copying reference data"

  for source_file in $test_bpe_data_dir/*.en; do
    file_name=${source_file##*/}
    dest_file=$bpe_data_dir/$file_name
    ln -sf $source_file $dest_file
  done

  for source_file in $test_raw_data_dir/*.fr; do
    file_name=${source_file##*/}
    dest_file=$bpe_data_dir/$file_name
    ln -sf $source_file $dest_file
  done

  ####
  # Translate using baseline
  ####

  log "Start translating (and scoring) with baseline"

  python /u/frenettx/maitrise/bt/couleurs/cli/evaluate/with_domain_embedding.py \
    calculate_scores \
    --out-dir=$evaluation_dir/bt \
    --domains-data-dir=$bpe_data_dir \
    --test-set-size=1000 \
    --test-set-bootstrap=1000 \
    --metrics=BLEU,chrF++ \
    --nb-bleu-scorers=$nb_scorer_processes \
    --nb-chrf-scorers=$nb_scorer_processes \
    --translation-model=$base_model_dir/checkpoint.avg10.pt \
    --translation-model-config=$model_config \
    --seed=$seed \
    --device=cuda \
    $@ >> $log_file

  ####
  # Translate using each WDE model
  ####

  for model_list in "${models[@]}"; do
    IFS=";" read -r -a model_data <<< "$model_list"
    model_name=${model_data[0]}
    model_dir=$base_wde_dir/${model_data[1]}
    model_translations_out=$evaluation_dir/$model_name
    other_args=""
    embedding_model=$base_model_dir/checkpoint.avg10.pt

    log "Start translating (and scoring) with model '$model_name'"

    mkdir -p $model_translations_out

    if [[ $model_name ==  "SentenceBERT" ]]; then
      model_config="/u/frenettx/maitrise/bt/experiments/with-domain-token/sentence-bert/model-config.json"
      other_args="$other_args
      --with-foreign-embeddings
      --embeddings-data-dir=$test_raw_data_dir"
    fi

    if [[ $model_name ==  "DDE" ]]; then
      embedding_model=/part/01/Tmp/frenettx/couleurs/with-domain-token/models/dde/train/final/best.ckpt
      other_args="$other_args
      --dde-dict-path=$base_model_dir/dict.en.txt
      --dde-batch-size=5120
      --dde-max-length=1024"
    fi

    python /u/frenettx/maitrise/bt/couleurs/cli/evaluate/with_domain_embedding.py \
      calculate_scores \
      --embedder=$model_name \
      --out-dir=$model_translations_out \
      --domains-data-dir=$bpe_data_dir \
      --embedding-set-sizes=100 \
      --embedding-set-bootstrap=25 \
      --test-set-size=1000 \
      --test-set-bootstrap=100 \
      --metrics=BLEU,chrF++ \
      --nb-bleu-scorers=$nb_scorer_processes \
      --nb-chrf-scorers=$nb_scorer_processes \
      --embedding-model=$embedding_model \
      --translation-model=$model_dir/data/checkpoint.avg10.pt \
      --translation-model-config=$model_config \
      --special-token="<dom-_X_>" \
      --seed=$seed \
      --device=cuda \
      $other_args \
      $@ >> $log_file
  done
fi

####
# Calculate BERTScore for the baseline
####

log "Start BERTScore for baseline"

python /u/frenettx/maitrise/bt/couleurs/cli/evaluate/with_domain_embedding.py \
  bertscore \
  --out-dir=$evaluation_dir/bt \
  --domains-data-dir=$bpe_data_dir \
  --no-embedder \
  --test-set-size=1000 \
  --test-set-bootstrap=1000 \
  --seed=$seed \
  --device=cuda \
  $@ >> $log_file

####
# Calculate BERTScore for each model
####

for model_list in "${models[@]}"; do
  IFS=";" read -r -a model_data <<< "$model_list"
  model_name=${model_data[0]}
  model_translations_out=$evaluation_dir/$model_name

  log "Start BERTScore for model '$model_name'"

  python /u/frenettx/maitrise/bt/couleurs/cli/evaluate/with_domain_embedding.py \
    bertscore \
    --out-dir=$model_translations_out \
    --domains-data-dir=$bpe_data_dir \
    --embedding-set-sizes=100 \
    --embedding-set-bootstrap=25 \
    --test-set-size=1000 \
    --test-set-bootstrap=100 \
    --seed=$seed \
    --device=cuda \
    $@ >> $log_file
done
