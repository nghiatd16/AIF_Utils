#!/bin/bash
log=logs/train.log
if [ $# -gt 0 ] && [ -f $1 ]; then
	log=$1
fi
header=$(cat $log | grep -v "ETA")
n=$(echo "$header" | wc -l)

echo "$header"
while true; do
	header=$(cat $log | grep -v "ETA")
	n2=$(echo "$header" | wc -l)
	if [ $n2 -ne $n ]; then
		n=$n2
		echo "$header"
	fi
	footer=$(cat $log | grep "ETA" | tail -1)
	echo -en "\r$footer     "
done
