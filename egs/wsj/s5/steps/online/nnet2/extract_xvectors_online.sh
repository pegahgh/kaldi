#!/bin/bash

# Copyright     2013  Daniel Povey
# Copyright     2016 Pegah Ghahremani
# Apache 2.0.


# This script extracts xVectors for a set of utterances, given
# features and a trained xVector extractor.

# The script is based on ^/egs/sre08/v1/sid/extract_ivectors.sh.  Instead of
# extracting a single iVector per utterance, it extracts one every few frames
# (controlled by the --ivector-period option, e.g. 10, which is to save compute).
# This is used in training (and not-really-online testing) of neural networks
# for online decoding.

# Rather than treating each utterance separately, it carries forward
# information from one utterance to the next, within the speaker.


# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
ivector_period=10
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.  Making this small during iVector
                    # extraction is equivalent to scaling up the prior, and will
                    # will tend to produce smaller iVectors where data-counts are
                    # small.  It's not so important that this match the value
                    # used when training the iVector extractor, but more important
                    # that this match the value used when you do real online decoding
                    # with the neural nets trained with these iVectors.
compress=true       # If true, compress the iVectors stored on disk (it's lossy
                    # compression, as used for feature matrices).
max_count=0         # The use of this option (e.g. --max-count 100) can make
                    # iVectors more consistent for different lengths of
                    # utterance, by scaling up the prior term when the
                    # data-count exceeds this value.  The data-count is after
                    # posterior-scaling, so assuming the posterior-scale is 0.1,
                    # --max-count 100 starts having effect after 1000 frames, or
                    # 10 seconds of data.

# End configuration section.

echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 [options] <data> <extractor-dir> <xvector-dir>"
  echo " e.g.: $0 data/train exp/xvector_sre exp/xvector_sre_a"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --xvector-period <int;default=10>                # How often to extract an xVector (frames)"
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

for f in $data/feats.scp $srcdir/final.raw; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log $dir/conf

xvector_model=$srcdir/final.raw

sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
#utils/split_data.sh $data $nj || exit 1;

echo $xvector_period > $dir/xvector_period || exit 1;
splice_opts=$(cat $srcdir/splice_opts)

# We need to create a config file for xVector extraction.
xeconf=$dir/conf/xvector_extractor.conf
echo -n >$xeconf
cp $srcdir/online_cmvn.conf $dir/conf/ || exit 1;
echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$xeconf
for x in $(echo $splice_opts); do echo "$x"; done > $dir/conf/splice.conf
echo "--xvector-period=$ivector_period" >>$xeconf
echo "--chunk-size=$chunk_size" >>$xeconf


absdir=$(readlink -f $dir)

for n in $(seq $nj); do
  # This will do nothing unless the directory $dir/storage exists;
  # it can be used to distribute the data among multiple machines.
  utils/create_data_link.pl $dir/xvector_online.$n.ark
done

if [ $stage -le 0 ]; then
  echo "$0: extracting xVectors"
  $cmd JOB=1:$nj $dir/log/extract_xvectors.JOB.log \
     nnet3-xvector-compute --config=$xeconf $xvector_model ark:$sdata/JOB/feat.scp ark:- \| \
     copy-feats --compress=$compress ark:- \
      ark,scp:$absdir/xvector_online.JOB.ark,$absdir/xvector_online.JOB.scp || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xVectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector_online.$j.scp; done >$dir/xvector_online.scp || exit 1;
fi
