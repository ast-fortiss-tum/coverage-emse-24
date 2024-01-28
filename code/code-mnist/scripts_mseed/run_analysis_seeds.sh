#!/bin/bash

# select seeds for analysis; use concrete seeds or an interval
# array=( 4307 6884 7474 7752 9298 5518 364 5874 8783 )
# for i in "${array[@]}"

for i in {100..200}
do
	python analysis.py -s "$i" -r 10
done
