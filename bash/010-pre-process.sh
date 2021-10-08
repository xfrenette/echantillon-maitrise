#!/bin/bash

set -e

####
# From the downloaded files, select a subset and preprocess it.
# From files downloaded by 000-download.sh, select for each domain a random
# subset of 7000 sentences. Remove identical source sentences, apply
# normalization, tokenization and apply BPE encoding.
####

base_dir=/part/01/Tmp/frenettx/couleurs/report-data/external-domains
bt_model_base_dir=/part/01/Tmp/frenettx/couleurs/baselines/trained_snmt/6jan2021_temperated/data
wmt_model_base_dir=/part/01/Tmp/frenettx/couleurs/baselines/pre_trained_snmt/model/wmt14.en-fr.joined-dict.transformer
data_dir=$base_dir/data
raw_dir=$data_dir/raw
log_file=$base_dir/log.txt
seed=1984
shuf=/u/frenettx/maitrise/bt/couleurs/bin/seeded_shuf.sh
fast_bpe=/part/01/Tmp/frenettx/opt/fastBPE/fast
langs=("en" "fr")
selection_size=10000


log()
{
  echo "$@" | tee -a $log_file
}

info()
{
  log "[$(date --rfc-3339=seconds)] $@"
}

error()
{
  log "[$(date --rfc-3339=seconds)] - ERROR - $@"
}

mkdir -p $base_dir $data_dir

touch $log_file

log "==========="
info "STARTED"
log "==========="

if [[ ! -d $raw_dir || ! $(ls -1 $raw_dir/* 2>/dev/null) ]]; then
  error "Could not find the downloaded data. Did you run 000-download.sh first?"
  exit 1
fi

####
# Filter out undesirable lines
####

# We first randomly select 3 times more sentences than what we need. We estimate
# that it will be sufficient that after filtering, we still have at least the
# number of desired sentences.

random_dir=$data_dir/01-random
filtered_dir=$data_dir/02-random-filtered
prefilter_selection_size=$((selection_size * 3))

mkdir -p $random_dir $filtered_dir

for sample_file in $raw_dir/*.${langs[0]}; do
  sample_file_name=${sample_file##*/}
  domain_name=${sample_file_name%%.*}

  for lang in "${langs[@]}"; do
    in_file=$raw_dir/$domain_name.$lang
    out_file=$random_dir/$domain_name.$lang

    if [[ -e $out_file ]]; then
      info "SKIPPING prefiltering $domain_name.$lang (already done)"
    else
      info "Randomly selecting $prefilter_selection_size sentences for $domain_name.$lang."
      cat $in_file | $shuf -s $seed | head -n $prefilter_selection_size > $out_file
    fi
  done
done

# We filter out duplicates and lines too short (ex: "6.")
if [ "$(ls -A $filtered_dir)" ]; then
  log "SKIPPING filtering, since the files seem to be already filtered " \
    "(empty the following directory to rerun: $filtered_dir)"
else
  log "Filtering the dataset by removing duplicates and too short sentences"
  python -u /u/frenettx/maitrise/bt/couleurs/cli/reports_scripts/filter_dataset.py \
    $random_dir \
    $filtered_dir | tee -a $log_file
fi

####
# Select random subset and preprocess (tokenize, normalize, ...)
####

# Select only top N lines in each filtered file
random_selected_dir=$data_dir/03-random-selected

mkdir -p $random_selected_dir

for sample_file in $filtered_dir/*.${langs[0]}; do
  sample_file_name=${sample_file##*/}
  domain_name=${sample_file_name%%.*}
  for lang in "${langs[@]}"; do
    in_file=$filtered_dir/$domain_name.$lang
    out_file=$random_selected_dir/$domain_name.$lang

    if [[ ! -e $out_file ]]; then
      head -n $selection_size $in_file > $out_file
    fi
  done
done

# Normalize and tokenize data files
normalized_dir=$data_dir/04-normalized
tokenized_dir=$data_dir/05-preprocessed
script=/u/frenettx/maitrise/bt/couleurs/cli/preprocessing.py

mkdir -p $normalized_dir $tokenized_dir

for lang in "${langs[@]}"; do
  # Normalize
  python -u $script normalize \
    --source_dir=$random_selected_dir \
    --file_pattern=*.$lang \
    --lang=$lang \
    --dest_dir=$normalized_dir | tee -a $log_file

  # Tokenize data files
  python -u $script tokenize \
    --source_dir=$normalized_dir \
    --file_pattern=*.$lang \
    --lang=$lang \
    --dest_dir=$tokenized_dir | tee -a $log_file
done

####
# Apply BPE
####

# Apply BPE (BT style)

bpe_bt_dir=$data_dir/06-bpe-bt
bpe_codes=$bt_model_base_dir/bpecodes
vocab_file=$bt_model_base_dir/dict.en.txt

mkdir -p $bpe_bt_dir

log "START applying BPE (BT style)"
for source_file in $tokenized_dir/*.{en,fr}; do
  file_name=${source_file##*/}
  dest_file=$bpe_bt_dir/$file_name
  stem=${file_name%.*}
  if [[ -e $dest_file ]]; then
    log "SKIPPING applying BPE to $file_name: already exists"
  else
    $fast_bpe applybpe $dest_file $source_file $bpe_codes $vocab_file
  fi
done

