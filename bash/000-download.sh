#!/bin/bash

set -e

####
# Download the different datasets, extract them and create one parallel corpus
# file (a raw text file) per domain. This script will convert TMX files to text
# files. But it won't do any preprocessing of the sentences.
####

# URLs of parallel datasets to download.
#
# Each line is a ";" separated list. First element of the line is the internal
# name of the domain. All following elements of the line are URLs of parallel
# data sets to download to combine in a single domain (most will have just one
# URL).
#
# For OPUS datasets, here is how to find the URL of a set. Go to the corpus'
# page on OPUS (ex: https://opus.nlpl.eu/GNOME-v1.php for the GNOME corpus) and
# in the "Statistics and TMX/Moses Downloads" table generally the second table),
# click on the language pair in the *lower left* triangle of the table (this
# triangle contains raw text files, and not TMX files). The link to put below is
# https://opus.nlpl.eu/download.php?f=GNOME/v1/moses/en-fr.txt.zip

parallel_data_urls=(
  "law;https://opus.nlpl.eu/download.php?f=JRC-Acquis/en-fr.txt.zip"
  "medical;https://opus.nlpl.eu/download.php?f=EMEA/v3/moses/en-fr.txt.zip"
  "bible;https://opus.nlpl.eu/download.php?f=bible-uedin/v1/moses/en-fr.txt.zip"
  "it;https://opus.nlpl.eu/download.php?f=PHP/v1/moses/en-fr.txt.zip;https://opus.nlpl.eu/download.php?f=GNOME/v1/moses/en-fr.txt.zip;https://opus.nlpl.eu/download.php?f=KDE4/v2/moses/en-fr.txt.zip;https://opus.nlpl.eu/download.php?f=Ubuntu/v14.10/moses/en-fr.txt.zip"
  "covid19;https://tico-19.github.io/data/TM/all.en-fr.tmx.zip"
)

base_dir=/part/01/Tmp/frenettx/couleurs/report-data/external-domains
data_dir=$base_dir/data
log_file=$base_dir/log.txt

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

####
# Download and extract files
####

download_dir=$data_dir/.downloads
out_dir=$data_dir/raw

mkdir -p $download_dir $out_dir

for url_infos in "${parallel_data_urls[@]}"; do
  IFS=";" read -r -a domain_data <<< "$url_infos"
  domain_name=${domain_data[0]}
  i=0
  base_out_file=$out_dir/$domain_name

  if [[ -e $base_out_file.en && -e $base_out_file.fr ]]; then
    info "Skipping $domain_name: already retrieved"
    continue
  fi

  rm -rf $base_out_file.en $base_out_file.fr $base_out_file.*.tmp

  for url in ${domain_data[@]:1}; do
    ext=${url##*.}
    download_file="$download_dir/$domain_name.$i.$ext"
    extract_dir="$download_dir/$domain_name.$i"
    extract_dir_tmp="$download_dir/.$domain_name.$i.tmp"

    if [[ "$ext" != "zip" ]]; then
      error "Only URLs that ends in .zip are supported ($url)"
      exit 1
    fi

    if [[ -d $extract_dir ]]; then
      info "Skipping downloading $url (already downloaded and extracted)"
    else
      # Downloading
      if [[ ! -e $download_file ]]; then
        info "Downloading file #$((i + 1)) for $domain_name ($url)"
        wget -O $download_file $url
      fi

      # Extracting
      rm -rf $extract_dir_tmp
      mkdir -p $extract_dir_tmp
      info "Extracting file #$((i + 1)) for $domain_name"
      unzip -d $extract_dir_tmp $download_file
      mv $extract_dir_tmp $extract_dir
    fi

    # If we have a single TMX file
    tmx_files=($extract_dir/*.tmx)
    if [[ ${#tmx_files[@]} != 0 ]]; then
      if [[ ${#tmx_files[@]} != 1 ]]; then
        error "More than one TMX file found. This is not supported for now."
      else
        tmx_file=${tmx_files[0]}
        info "Processing TMX file: $tmx_file"
        python -u /u/frenettx/maitrise/bt/couleurs/cli/reports_scripts/extract_tmx.py \
          $tmx_file \
          $out_dir \
          --out-pattern=$domain_name.{lang}
      fi
    else
      for lang in "en" "fr"; do
        out_file=$base_out_file.$lang
        tmp_out_file=$out_file.tmp

        touch $tmp_out_file
        cat $extract_dir/*.$lang >> $tmp_out_file
      done
    fi

    i=$((i + 1))
  done

  for lang in "en" "fr"; do
    tmp_file=$base_out_file.$lang.tmp

    if [[ -e $tmp_file ]]; then
      mv $base_out_file.$lang.tmp $base_out_file.$lang
    fi
  done
done

