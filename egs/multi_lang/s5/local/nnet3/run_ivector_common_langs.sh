#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
pitch_conf=conf/pitch.conf
use_sep_init_layer=false
voicing_conf=

[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

# bengali: "conf/lang/103-bengali-limitedLP.official.conf"
# assamese: "conf/lang/102-assamese-limitedLP.official.conf"
# cantonese: "conf/lang/101-cantonese-limitedLP.official.conf"
# pashto: "conf/lang/104-pashto-limitedLP.official.conf"
# tagalog: "conf/lang/106-tagalog-limitedLP.official.conf"
# turkish: "conf/lang/105-turkish-limitedLP.official.conf"
# vietnamese: "conf/lang/107-vietnamese-limitedLP.official.conf"
# haitian: "conf/lang/201-haitian-limitedLP.official.conf"
# lao: "conf/lang/203-lao-limitedLP.official.conf"
# zulu: "conf/lang/206-zulu-limitedLP.official.conf"
# tamil: "conf/lang/204-tamil-limitedLP.official.conf"
L=$1

case "$L" in
		BNG)
			langconf=conf/lang/103-bengali-limitedLP.official.conf
			;;
		ASM)			
			langconf=conf/lang/102-assamese-limitedLP.official.conf
			;;
		CNT)
			langconf=conf/lang/101-cantonese-limitedLP.official.conf
			;;
		PSH)
			langconf=conf/lang/104-pashto-limitedLP.official.conf
			;;
		TGL)
			langconf=conf/lang/106-tagalog-limitedLP.official.conf
			;;
		TUR)
			langconf=conf/lang/105-turkish-limitedLP.official.conf	
			;;
		VTN)
			langconf=conf/lang/107-vietnamese-limitedLP.official.conf
			;;
		HAI)
			langconf=conf/lang/201-haitian-limitedLP.official.conf
			;;
		LAO)
			langconf=conf/lang/203-lao-limitedLP.official.conf
			;;
		ZUL)
			langconf=conf/lang/206-zulu-limitedLP.official.conf	
			;;
		TAM)
			langconf=conf/lang/204-tamil-limitedLP.official.conf	
			;;
    SWBD)
      langconf=conf/lang/swbd.conf
      ;;
		*)
			echo "Unknown language code $L." && exit 1
esac

mkdir -p langconf/$L
rm -rf langconf/$L/*
cp $langconf langconf/$L/lang.conf
langconf=langconf/$L/lang.conf

[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
. $langconf || exit 1;

mkdir -p nnet3
# perturbed data preparation
train_set=train
if [ "$speed_perturb" == "true" ]; then
  train_set=train_sp
fi

if [ "$use_sep_init_layer" == "true" ];then
  extractor=exp/$L/nnet3/extractor
  ivector_suffix=
else
  extractor=exp/multi/nnet3/extractor
  ivector_suffix=_gb
fi

echo ivector_suffix = $ivector_suffix
if $use_sep_init_layer; then
  # ivector extractor training
  if [ $stage -le 5 ]; then
    # We need to build a small system just because we need the LDA+MLLT transform
    # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
    # the transform (12th iter is the last), any further training is pointless.
    # this decision is based on fisher_english
    mkdir -p exp/$L/nnet3
    steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
      --splice-opts "--left-context=3 --right-context=3" \
      --boost-silence $boost_sil \
      $numLeavesMLLT $numGaussMLLT data/$L/${train_set}_hires \
      data/$L/lang exp/$L/tri5_ali_sp exp/$L/nnet3/tri3b
  fi

  if [ $stage -le 6 ]; then
    # To train a diagonal UBM we don't need very much data, so use the smallest subset.
    steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
      data/$L/${train_set}_hires 512 exp/$L/nnet3/tri3b exp/$L/nnet3/diag_ubm
  fi

  if [ $stage -le 7 ]; then
    # iVector extractors can be sensitive to the amount of data, but this one has a
    # fairly small dim (defaults to 100) so we don't use all of it, we use just the
    # 100k subset (just under half the data).
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
      data/$L/${train_set}_hires exp/$L/nnet3/diag_ubm exp/$L/nnet3/extractor || exit 1;
  fi
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/$L/${train_set}_hires data/$L/${train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/$L/${train_set}_max2_hires $extractor exp/$L/nnet3/ivectors_${train_set}${ivector_suffix} || exit 1;

fi


exit 0;
