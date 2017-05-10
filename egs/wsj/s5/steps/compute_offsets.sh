#!/bin/bash

# Copyright 2016 Pegah Ghahremani
# Apache 2.0

# Computes random offsets per speaker w.r.t speakers's covariance matrix.
# It first computes mean per speaker and global mean.

# Begin configuration section.
cmd=
offsets_config=conf/offsets.conf
compress=true
ubm_offset=
nj=30
offset_type=0 # 0 : same offsets for all frames in speaker
              # 1 : per-frame offset for each frame using ubm posterior.
# End configuration section. 

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ];then 
  echo "Usage: $0 [opts] <data-dir> <log-dir> <offset-dir>"
  echo " e.g.: $0 data/train exp/make_offset/train offsets"
  exit 1;
fi

data=$1
logdir=$2
offsetdir=$3


# make $offsetdir an absolute pathname.
offsetdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $offsetdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $offsetdir || exit 1;
mkdir -p $logdir || exit 1;


required_files="$data/feats.scp $data/spk2utt $offsets_config"

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done
echo $offset_type > $data/offset_type
if [ $offset_type -eq 0 ]; then
  $cmd $logdir/compute_rand_offset.log \ 
    generate-random-cmn-offsets --config=$offsets_config ark:$data/spk2utt scp:$data/feats.scp ark:- | copy-feats --compress=$compress ark:- ark,scp:$offsetdir/offsets.ark,$data/offsets.scp || exit 1;
elif [ $offset_type -eq 1 ]; then
  $cmd JOB=1:$nj $logdir/compute_rand_offset.JOB.log \
    generate-random-per-frame-offsets --config=$offsets_config $ubm_offset \
    ark:$sdata/JOB/spk2utt scp:$sdata/JOB/feats.scp ark:- \| \
    copy-feats --compress=$compress ark:- ark,scp:$offsetdir/offsets.JOB.ark,$offsetdir/offsets.JOB.scp || exit 1;

  for n in $(seq $nj);do
    cat $offsetdir/offsets.$n.scp ||  exit 1;
  done > $data/offsets.scp

fi
echo "Succeeded creating random offsets for $name"
