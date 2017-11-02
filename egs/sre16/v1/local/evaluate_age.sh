#!/bin/bash

# Copyright 2017 Pegah Ghahremani
# Apache 2.0

# This script evaluates age detection model trained using nnet3 model.

[ -f ./path.sh ] && . ./path.sh; # source the path.

. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <data-dir> <xvector-data>";
  echo "e.g.: $0 data/sre_08_10/1/test exp/xvector_age/1/test"
fi

data=$1
xvector_data=$2

for x in $xvector_data/spk_xvector.scp $xvector_data/xvector.scp $data/utt2age; do
  [ ! -f $x ] && echo "$0: file $x does not exist" && exit 1;
done

copy-vector scp:$xvector_data/xvector.scp ark,t:- | \
  awk '{max=$3; for (i=3;i<NF;i++) if($i>max) {max=$i;m_ind=i-3 };print $1,max,m_ind}' > $xvector_data/utt2max

age_offset=`cat $data/age_offset` || exit 1;

mean_abs_error=`paste <(awk '{print $3}' $xvector_data/utt2max) <(awk -v offset=$age_offset '{print $2-offset}' $data/utt2age) > $xvector_data/true.precicted.age
cat $xvector_data/true.precicted.age | awk '{ sum +=( $1-$2 >= 0 ? $1-$2 : $2-$1 ); n++ } END { if (n > 0) print sum / n; }'`

echo "mean abs error |y_i - y'_i| is $mean_abs_error." > $xvector_data/mean_abs_error.txt
