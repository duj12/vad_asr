#!/bin/bash

stage=3
input_wav_path=data/input/
wav_list_path=data/wav.list
output_wav_path=data/output

# step 1. put long wavs to data/input, cut them and output in data/output
if [ $stage -le 1 ]; then
  find $input_wav_path -name "*.wav" > $wav_list_path
  if [ -d $output_wav_path ]; then
    rm -r $output_wav_path
  fi
  mkdir -p $output_wav_path

  python vad_cut.py $wav_list_path $output_wav_path/wav
fi

# step 2. Decoding short wavs.
if [ $stage -le 2 ]; then
  find `pwd`/$output_wav_path  -name "*.wav" | \
  awk -F"/" -v name="" '{name=$NF; gsub(".wav","",name); print name" "$0}' | \
  sort > $output_wav_path/wav.scp

  bash ./infer.sh $output_wav_path $output_wav_path
fi

# step3. Remove the spaces between Chinese chars
if [ $stage -le 3 ]; then
  python utils/remove_space_between_chinese.py  \
      $output_wav_path/1best_recog/text $output_wav_path/asr.txt
fi