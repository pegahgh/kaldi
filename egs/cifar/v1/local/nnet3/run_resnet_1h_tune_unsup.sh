#!/bin/bash
# _1h_tune_unsup is as _1h_tune, but it combines training and test data where
# weight for test data is 0.0 and the the average of outputs for two pertubred
# versions of test example is used as label during training.

# 1e is as 1d but with more filters and epochs.

# local/nnet3/compare.sh exp/resnet1d_cifar10/ exp/resnet1e_cifar10/
# System                resnet1d_cifar10 resnet1e_cifar10
# final test accuracy:       0.9537      0.9583
# final train accuracy:       0.9966      0.9994
# final test objf:         -0.139607   -0.124945
# final train objf:       -0.0219607 -0.00603407
# num-parameters:           1322730     3465194

# local/nnet3/compare.sh exp/resnet1d_cifar100 exp/resnet1e_cifar100
# System                resnet1d_cifar100 resnet1e_cifar100
# final test accuracy:       0.7687      0.7914
# final train accuracy:       0.9276     0.9922
# final test objf:         -0.812203   -0.786857
# final train objf:        -0.265734   -0.0514912
# num-parameters:           1345860     3511364
# steps/info/nnet3_dir_info.pl exp/resnet1c_cifar10{,0}
# exp/resnet1e_cifar10: num-iters=186 nj=1..2 num-params=3.5M dim=96->10 combine=-0.01->-0.01 loglike:train/valid[123,185,final]=(-0.109,-0.026,-0.0060/-0.21,-0.167,-0.125) accuracy:train/valid[123,185,final]=(0.963,0.9936,0.9994/0.930,0.949,0.958)
# exp/resnet1e_cifar100/: num-iters=186 nj=1..2 num-params=3.5M dim=96->100 combine=-0.09->-0.07 loglike:train/valid[123,185,final]=(-0.53,-0.109,-0.051/-1.06,-0.93,-0.79) accuracy:train/valid[123,185,final]=(0.844,0.9730,0.9922/0.713,0.760,0.791)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
dataset=cifar100
srand=0
reporting_email=
affix=1h
num_rotation_classes=12
lrate_skd=exp
k=12
epochs=200
unsup_regularizer=0.5
use_unsup=true
dir_affix=2
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

affix=${affix}_${lrate_skd}_k${k}_${epochs}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi



dir=exp/resnet${affix}_${dataset}_unsup${dir_affix}

#egs=exp/${dataset}_egs2
egs=exp/${dataset}_egs_weighted
if [ $stage -le 0 ] && $use_unsup; then
# prepare weighted eg for training using train+test data, where test data
# trained using kl-divergence.
  if false; then # 100
  image/copy_data_dir.sh --image-prefix train- data/cifar100_train data/cifar100_train_v2
  image/copy_data_dir.sh --image-prefix test- data/cifar100_test data/cifar100_test_v2
  image/nnet3/get_egs.sh --cmd "$train_cmd" \
    --generate-egs-scp true data/cifar100_test_v2 data/cifar100_test_v2 \
    exp/cifar100_egs_test || exit 1;
  image/nnet3/get_egs.sh --cmd "$train_cmd" \
    --generate-egs-scp true \
    data/cifar100_train_v2 data/cifar100_test_v2 \
    exp/cifar100_egs_train || exit 1;
  fi
# weighted combination of test and train egs.
  egs_opts="--lang2weight 1,0"
  steps/nnet3/multilingual/combine_egs.sh $egs_opts \
    --cmd "$train_cmd" --samples-per-iter 30000 \
    2 exp/cifar100_egs_train exp/cifar100_egs_test $egs || exit 1;
  rm $egs/egs.output.*.ark
  mv $egs/valid_diagnostic.output.ark $egs/valid_diagnostic.output.ark.bkup
  mv $egs/train_diagnostic.output.ark $egs/train_diagnostic.output.ark.bkup
  for f in feat_dim frames_per_eg left_context right_context output_dim;do
    cp -r exp/cifar100_egs_train/info/$f $egs/info/.
  done
fi

if [ ! -d $egs ]; then
  echo "$0: expected directory $egs to exist.  Run the get_egs.sh commands in the"
  echo "    run.sh before this script."
  exit 1
fi

# check that the expected files are in the egs directory.
egs_files_to_check="$egs/egs.1.ark $egs/train_diagnostic.egs $egs/valid_diagnostic.egs $egs/combine.egs"
if $use_unsup; then
  egs_files_to_check="$egs/egs.1.scp $egs/train_diagnostic.scp $egs/valid_diagnostic.scp $egs/combine.scp"
fi
for f in $egs_files_to_check \
         $egs/info/feat_dim $egs/info/left_context $egs/info/right_context \
         $egs/info/output_dim; do
  if [ ! -e $f ]; then
    echo "$0: expected file $f to exist."
    exit 1;
  fi
done


mkdir -p $dir/log


if [ $stage -le 1 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(cat $egs/info/output_dim)
  num_new_targets=$[num_targets*num_rotation_classes]

  # Note: we hardcode in the CNN config that we are dealing with 32x3x color
  # images.

  nf1=$((16 * $k))
  nf2=$(($nf1 * 2))
  nf3=$(($nf2 * 2))
  nb3=$nf2

  a="num-minibatches-history=40.0"
  common="$a required-time-offsets=0 height-offsets=-1,0,1"
  res_opts="$a bypass-source=batchnorm"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=96 name=input
  conv-layer name=conv1 $a height-in=32 height-out=32 time-offsets=-1,0,1 required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=$nf1
  res-block name=res2 num-filters=$nf1 height=32 time-period=1 $res_opts
  res-block name=res3 num-filters=$nf1 height=32 time-period=1 $res_opts
  conv-layer name=conv4 height-in=32 height-out=16 height-subsample-out=2 time-offsets=-1,0,1 $common num-filters-out=$nf2
  res-block name=res5 num-filters=$nf2 height=16 time-period=2 $res_opts
  res-block name=res6 num-filters=$nf2 height=16 time-period=2 $res_opts
  conv-layer name=conv7 height-in=16 height-out=8 height-subsample-out=2 time-offsets=-2,0,2 $common num-filters-out=$nf3
  res-block name=res8 num-filters=$nf3 num-bottleneck-filters=$nb3 height=8 time-period=4 $res_opts
  res-block name=res9 num-filters=$nf3 num-bottleneck-filters=$nb3 height=8 time-period=4 $res_opts
  res-block name=res10 num-filters=$nf3 num-bottleneck-filters=$nb3 height=8 time-period=4 $res_opts
  channel-average-layer name=channel-average input=Append(2,6,10,14,18,22,24,28) dim=$nf3
  output-layer name=output learning-rate-factor=0.1 dim=$num_targets
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  echo "output-node name=output-unsup input=output.log-softmax objective=linear" >> $dir/configs/final.config
fi


if [ $stage -le 2 ]; then

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --image.augmentation-opts="--horizontal-flip-prob=0.5 --horizontal-shift=0.1 --vertical-shift=0.1 --num-rotation-classes=$num_rotation_classes --rotation-class-prob=0.5 --num-channels=3" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$epochs \
    --trainer.lrate-skd=$lrate_skd \
    --egs.frames-per-eg=1 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.003 \
    --trainer.optimization.final-effective-lrate=0.0003 \
    --trainer.optimization.minibatch-size=64,32,16 \
    --trainer.optimization.proportional-shrink=50.0 \
    --trainer.shuffle-buffer-size=2000 \
    --trainer.unsup-regularizer=$unsup_regularizer \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
