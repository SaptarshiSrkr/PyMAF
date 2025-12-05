#!/bin/bash

for samples in 10 50 100 500 1000 5000 10000 50000
do
    # moving average
    python moving_average.py $samples
    # KL divergence
    python KL_div.py $samples
done
wait

# for samples in 50000 10000 5000 1000 500 100 50 10 5
# do
# done
# wait
