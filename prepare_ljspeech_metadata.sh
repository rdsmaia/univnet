#!/bin/bash

if [ $# -ne 1 ]; then
    echo "usage: prepare_ljspeech_metadata.sh path_to_LJSPeech-1.1"
    exit
fi
ljpath=$1
echo $ljpath

#sed -e "s|LJ0|${ljpath}LJ0|g" ${ljpath}/metadata.csv | awk -F "|" '{$3="0";$1=$1".wav";OFS="|";print $0}' | sort -R > shuff
awk -F "|" '{OFS="|"}{$3="0";print $0}' ${ljpath}/metadata.csv | sort -R > shuff
head -n 13000 shuff > ${ljpath}/ljspeech_train.tsv
tail -n   100 shuff > ${ljpath}/ljspeech_val.tsv
rm -f shuff
