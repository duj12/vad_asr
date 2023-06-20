#!/bin/bash

find `pwd`/data -name "*.wav" > wav.list

output_dir=output
mkdir -p $output_dir
python forward.py -l wav.list -soft -o $output_dir    #需要保存切分的wav和vad后验概率和波形图时，使用-o选项