#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
chunk_size=150
xvector_period=50
xvector_model=exp/xvectors_sre_a
xvector_dir=exp/nnet3
extract_type=0
suffix=_hires

. ./path.sh
. ./utils/parse_options.sh

mkdir -p nnet3
# perturbed data preparation
train_set=train_nodup
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed

    for datadir in train_nodup; do
      utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
      utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
      utils/combine_data.sh data/${datadir}_tmp data/temp1 data/temp2
      utils/validate_data_dir.sh --no-feats data/${datadir}_tmp
      rm -r data/temp1 data/temp2

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
        data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_tmp

      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
      utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
      utils/fix_data_dir.sh data/${datadir}_sp
      rm -r data/temp0 data/${datadir}_tmp
    done
  fi

  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/train_nodup_sp data/lang_nosp exp/tri4 exp/tri4_ali_nodup_sp || exit 1
  fi
  train_set=train_nodup_sp
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc${suffix}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  # the 100k_nodup directory is copied seperately, as
  # we want to use exp/tri2_ali_100k_nodup for lda_mllt training
  # the main train directory might be speed_perturbed
  for dataset in $train_set train_100k_nodup; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}${suffix}

    # scale the waveforms, this is useful as we don't use CMVN
    data_dir=data/${dataset}${suffix}
    cat $data_dir/wav.scp | python -c "
import sys, os, subprocess, re, random
scale_low = 1.0/8
scale_high = 2.0
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), random.uniform(scale_low, scale_high))
"| sort -k1,1 -u  > $data_dir/wav.scp_scaled || exit 1;
    mv $data_dir/wav.scp_scaled $data_dir/wav.scp

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc${suffix}.conf \
        --cmd "$train_cmd" data/${dataset}${suffix} exp/make${suffix}/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}${suffix} exp/make${suffix}/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}${suffix};
  done

  for dataset in eval2000 train_dev rt03; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}${suffix}
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc${suffix}.conf \
        data/${dataset}${suffix} exp/make${suffix}/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}${suffix} exp/make${suffix}/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}${suffix}  # remove segments with problems
  done

fi

if [ $stage -le 4 ]; then
  # We extract xVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (xVector starts at zero).
  # steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires data/${train_set}_max2_hires
  if [ ! -f $xvector_dir/xvectors_$train_set/.xvector.done ]; then
    echo extract_type inside = $extract_type
    steps/nnet3/extract_xvectors.sh --xvector-period $xvector_period \
      --chunk-size $chunk_size \
      --extract-type $extract_type \
      --cmd "$train_cmd" --nj 4 \
      data/${train_set}${suffix} $xvector_model $xvector_dir/xvectors_$train_set || exit 1;
    touch $xvector_dir/xvectors_$train_set/.xvector.done
  fi
  for data_set in eval2000 train_dev rt03; do
    if [ ! -f  $xvector_dir/xvectors_$data_set/.xvector.done ]; then
      steps/nnet3/extract_xvectors.sh --xvector-period $xvector_period \
        --chunk-size $chunk_size \
        --extract-type $extract_type \
        --cmd "$train_cmd" --nj 4 \
        data/${data_set}${suffix} $xvector_model $xvector_dir/xvectors_$data_set || exit 1;
      touch $xvector_dir/xvectors_$data_set/.xvector.done
    fi
  done
fi

exit 0;
