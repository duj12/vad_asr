#!/bin/bash

find `pwd`/data -name "*.wav" > wav.list

output_dir=output_stream
mkdir -p $output_dir
python vad_models.py