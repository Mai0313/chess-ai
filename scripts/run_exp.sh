#!/bin/bash

experiments=("md1" "md2" "md3" "md4" "md5" "md6" "md7" "md8")

for experiment in "${experiments[@]}"; do
    python src/train.py experiment=${experiment}
done
