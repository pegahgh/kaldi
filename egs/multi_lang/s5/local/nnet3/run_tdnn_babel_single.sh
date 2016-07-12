#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
speed_perturb=true
max_change=2.0
max_shift=0.0
affix=
splice_indexes="-2,-1,0,1,2 0 -2,2 0 -7,2 0" 
#splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 0"
common_egs_dir=
relu_dim=400
num_epochs=4
remove_egs=false
aux_suffix=_mfcc
ivector_suffix=
basename=tdnn_v3
##

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

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $# -ne 1 ]; then
  echo "Usage: $(basename $0) [opts] <lang>"
  echo " e.g.: $(basename $0)  ASM"
  exit 1
fi

L=$1

data=data/$L
dir=exp/$L/nnet3/$basename
alidir=exp/$L/tri5_ali

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
		*)
			echo "Unknown language code $L." && exit 1
esac

mkdir -p langconf/$L
rm -rf langconf/$L/*
cp $langconf langconf/$L/lang.conf
langconf=langconf/$L/lang.conf


[ ! -f $langconf ] && echo 'Language configuration $langconf does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
. $langconf || exit 1;

[ -f local.conf ] && . local.conf;

echo using "Language = $L, config = $langconf"

##


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

train_nj=72
export boost_sil use_pitch train_nj
export numLeavesMLLT numGaussMLLT


suffix=
affix=

if [ "$speed_perturb" == "true" ]; then
  suffix=${suffix}_sp
fi

if $use_pitch;then aux_suffix=${aux_suffix}_pitch ; fi
if $use_entropy;then aux_suffix=${aux_suffix}_entropy ; fi

echo "use_pitch = $use_pitch , use_entropy = $use_entropy aux_suffix = $aux_suffix"
dir=$dir${affix:+_$affix}
dir=${dir}${suffix}${aux_suffix}
train_set=train$suffix
alidir=${alidir}${suffix}

#local/nnet3/run_ivector_common_v2.sh --stage $stage \
#	--speed-perturb $speed_perturb || exit 1;

if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  echo use-ivector = $use_ivector  
  if $use_ivector; then
    ivector_opts="--online-ivector-dir exp/$L/nnet3/ivectors_${train_set}${ivector_suffix}"
  fi

  steps/nnet3/train_tdnn.sh --stage $train_stage --remove-egs false \
    --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 6 \
    --splice-indexes "$splice_indexes" --max-param-change $max_change \
    --feat-type raw --max-shift $max_shift $ivector_opts \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.0017 --final-effective-lrate 0.00017 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --relu-dim $relu_dim \
    --remove-egs $remove_egs \
    $data/${train_set}_hires${aux_suffix} $data/lang $alidir $dir  || exit 1;
  touch $dir/.done
fi


exit 0;

