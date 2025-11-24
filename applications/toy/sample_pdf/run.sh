#!/bin/bash

# moving average
for samples in 50000 10000 5000 1000 500 100 50 10 5
do
    python moving_average.py $samples &
done
wait

# KL divergence
for samples in 50000 10000 5000 1000 500 100 50 10 5
do
    python KL_div.py $samples &
done
wait
