#!/bin/bash

array=( 4226 4381 129 7352 351 3788 1258 7315 8185 8783 240)

for i in "${array[@]}"
do
	python run.py -e 10 -s "$i"
done
