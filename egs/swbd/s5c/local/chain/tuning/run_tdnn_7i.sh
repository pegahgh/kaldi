#!/bin/bash
# _7i is as _7h but it uses sibling network and multi-stage training
# to transfer information from larger network to smaller network.
# It uses multi-stage training to train sibling network with smaller parameter.
# The 1st stage of training is as as basleline tdnn_7d and it trains primary network.
# The second stage of training is to use regularizers in all layers as objectives
# to train sibling network and in the 3rd stage, we train a sibling network using
# chain objective for 1 epoch.

#System                  tdnn_7g   tdnn_7h
#WER on train_dev(tg)      13.98     13.84
#WER on train_dev(fg)      12.78     12.84
#WER on eval2000(tg)        16.7      16.5
#WER on eval2000(fg)        14.9      14.8
#Final train prob     -0.0817467-0.0889771
#Final valid prob      -0.110475 -0.113102
#Final train prob (xent)      -1.20065   -1.2533
#Final valid prob (xent)       -1.3313  -1.36743
#
set -e

# configs for 'chain'
affix=
stage=12
multi_stage_train=1
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_7i  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
sibling_initial_effective_lrate=0.0005
sibling_final_effective_lrate=0.00005
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
frames_per_eg=150
remove_egs=false
common_egs_dir=exp/chain/tdnn_7h_sp/egs
xent_regularize=0.1
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_7d_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser for primary network";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  mkdir -p $dir/stage1
  mkdir -p $dir/stage1/configs
  cat <<EOF > $dir/stage1/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(input@-1,input,input@1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/stage1/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=625
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=625

  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain input=tdnn7 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-renorm-layer name=prefinal-xent input=tdnn7 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/stage1/configs/network.xconfig \
    --config-dir $dir/stage1/configs/

  echo "$0: creating neural net configs using the xconfig parser for sibling network"
  sibling_dim=300
  primary_dim=625
  regressor_lr_factor=1.0
  regressor_scale=`echo $regressor_lr_factor $primary_dim | awk '{printf "%.8f \n", $1/$2}'`
  regressor_scale_vec=""
  for i in `seq $primary_dim`;do
    regressor_scale_vec="$regressor_scale_vec $regressor_scale"
  done

  mkdir -p $dir/stage2
  cat <<EOF > $dir/stage2/regressor_scale.vec
  [ $regressor_scale_vec ]
EOF

  mkdir -p $dir/stage2/configs
  cat <<EOF > $dir/stage2/configs/network.xconfig

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1-sibling input=lda dim=$sibling_dim
  relu-renorm-layer name=tdnn2-sibling input=Append(-1,0,1) dim=$sibling_dim
  relu-renorm-layer name=tdnn3-sibling input=Append(-1,0,1) dim=$sibling_dim
  relu-renorm-layer name=tdnn4-sibling input=Append(-3,0,3) dim=$sibling_dim
  relu-renorm-layer name=tdnn5-sibling input=Append(-3,0,3) dim=$sibling_dim
  relu-renorm-layer name=tdnn6-sibling input=Append(-3,0,3) dim=$sibling_dim
  relu-renorm-layer name=tdnn7-sibling input=Append(-3,0,3) dim=$sibling_dim

  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain-sibling input=tdnn7-sibling dim=$sibling_dim target-rms=0.5
  output-layer name=output-sibling include-log-softmax=false dim=$num_targets max-change=1.5

  relu-renorm-layer name=prefinal-xent-sibling input=tdnn7-sibling dim=$sibling_dim target-rms=0.5
  output-layer name=output-xent-sibling dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

  ## adding the regressor outputs to the sibling network configs.
  relu-renorm-layer name=tdnn2-regressor input=tdnn2-sibling dim=$primary_dim
  regressor-layer name=regressor-2 input1=tdnn2-regressor input2=tdnn2 objective-type=linear max-change=1.5 dim=$primary_dim regressor-scale-file=$dir/stage2/regressor_scale.vec supervision-type=unsupervised

  relu-renorm-layer name=tdnn3-regressor input=tdnn3-sibling dim=$primary_dim
  regressor-layer name=regressor-3 input1=tdnn3-regressor input2=tdnn3 objective-type=linear max-change=1.5 dim=$primary_dim regressor-scale-file=$dir/stage2/regressor_scale.vec supervision-type=unsupervised

  relu-renorm-layer name=tdnn4-regressor input=tdnn4-sibling dim=$primary_dim
  regressor-layer name=regressor-4 input1=tdnn4-regressor input2=tdnn4 objective-type=linear max-change=1.5 dim=$primary_dim regressor-scale-file=$dir/stage2/regressor_scale.vec supervision-type=unsupervised

  relu-renorm-layer name=tdnn5-regressor input=tdnn5-sibling dim=$primary_dim
  regressor-layer name=regressor-5 input1=tdnn5-regressor input2=tdnn5 objective-type=linear max-change=1.5 dim=$primary_dim regressor-scale-file=$dir/stage2/regressor_scale.vec supervision-type=unsupervised

  relu-renorm-layer name=tdnn6-regressor input=tdnn6-sibling dim=$primary_dim
  regressor-layer name=regressor-6 input1=tdnn6-regressor input2=tdnn6 objective-type=linear max-change=1.5 dim=$primary_dim regressor-scale-file=$dir/stage2/regressor_scale.vec supervision-type=unsupervised

  relu-renorm-layer name=tdnn7-regressor input=tdnn7-sibling dim=$primary_dim
  regressor-layer name=regressor-7 input1=tdnn7-regressor input2=tdnn7 objective-type=linear max-change=1.5 dim=$primary_dim regressor-scale-file=$dir/stage2/regressor_scale.vec supervision-type=unsupervised
EOF
  steps/nnet3/xconfig_to_configs.py --aux-xconfig-file $dir/stage1/configs/network.xconfig \
    --xconfig-file $dir/stage2/configs/network.xconfig --config-dir $dir/stage2/configs/

  # we skip add_compatiblity stage in xconfig_to_config.py
  # we copy vars from stage1 to stage2 and stage3 for now.
  cp -r $dir/stage1/configs/vars $dir/stage2/configs/.
  cp -r $dir/stage1/configs/vars $dir/stage3/configs/.
  # edits.config contains edits required for different stage of training.
  # it is applied to 0.mdl generated at prepare_initial_network stage in
  # iter -1.
  # The edits for 2nd stage contains renaming primary network's outputs to
  # <output-name>-primary to not train using these outputs.
  # the edits contain renaming sibling network output to be output.
  cat <<EOF > $dir/stage2/configs/edits.config
  rename-node old-name=output new-name=output-primary
  rename-node old-name=output-xent new-name=output-xent-primary
  rename-node old-name=output-sibling new-name=output
  rename-node old-name=output-xent-sibling new-name=output-xent
EOF
  # edits.config contains edits required for 3rd stage of training.
  mkdir -p $dir/stage3
  mkdir -p $dir/stage3/configs
  cat <<EOF > $dir/stage3/configs/edits.config
  remove-output-nodes name=regressor*
  remove-output-nodes name=*-primary
  remove-orphans
EOF
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi
  if [ $multi_stage_train -le 0 ] && [ ! -f $dir/stage1/final.mdl ]; then
    echo "$0: Training primary network"
    steps/nnet3/chain/train.py --stage $train_stage \
      --cmd "$decode_cmd" \
      --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
      --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
      --chain.xent-regularize $xent_regularize \
      --chain.leaky-hmm-coefficient 0.1 \
      --chain.l2-regularize 0.00005 \
      --chain.apply-deriv-weights false \
      --chain.lm-opts="--num-extra-lm-states=2000" \
      --egs.dir "$common_egs_dir" \
      --egs.stage $get_egs_stage \
      --egs.opts "--frames-overlap-per-eg 0" \
      --egs.chunk-width $frames_per_eg \
      --trainer.num-chunk-per-minibatch $minibatch_size \
      --trainer.frames-per-iter 1500000 \
      --trainer.num-epochs $num_epochs \
      --trainer.optimization.num-jobs-initial $num_jobs_initial \
      --trainer.optimization.num-jobs-final $num_jobs_final \
      --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
      --trainer.optimization.final-effective-lrate $final_effective_lrate \
      --trainer.max-param-change $max_param_change \
      --cleanup.remove-egs false \
      --feat-dir data/${train_set}_hires \
      --tree-dir $treedir \
      --lat-dir exp/tri4_lats_nodup$suffix \
      --dir $dir/stage1  || exit 1;
  fi

  if [ $multi_stage_train -le 1 ]; then
      mkdir -p $dir/stage2
      echo "$0: copy final primary network in $dir/stage1/final.raw to "
      echo "$dir/stage2/init.raw as initial network with zero lr factor"
      echo "as primary network for sibling network."
      nnet3-am-copy --raw=true \
        --edits='set-learning-rate-factor name=* learning-rate-factor=0.0;' \
        $dir/stage1/final.mdl $dir/stage2/init.raw || exit 1;

      echo "$0: Training sibling network using regularizer objectives."
      steps/nnet3/chain/train.py --stage $train_stage \
      --cmd "$decode_cmd" \
      --init-raw-model $dir/stage2/init.raw \
      --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
      --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
      --chain.leaky-hmm-coefficient 0.1 \
      --chain.l2-regularize 0.0 \
      --chain.chain-regularize 0.00000001 \
      --chain.apply-deriv-weights false \
      --chain.lm-opts="--num-extra-lm-states=2000" \
      --egs.dir "$common_egs_dir" \
      --egs.stage $get_egs_stage \
      --egs.opts "--frames-overlap-per-eg 0" \
      --egs.chunk-width $frames_per_eg \
      --trainer.num-chunk-per-minibatch $minibatch_size \
      --trainer.frames-per-iter 1500000 \
      --trainer.num-epochs 1 \
      --trainer.optimization.num-jobs-initial $num_jobs_initial \
      --trainer.optimization.num-jobs-final $num_jobs_final \
      --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
      --trainer.optimization.final-effective-lrate $final_effective_lrate \
      --trainer.max-param-change $max_param_change \
      --cleanup.remove-egs false \
      --feat-dir data/${train_set}_hires \
      --tree-dir $treedir \
      --lat-dir exp/tri4_lats_nodup$suffix \
      --dir $dir/stage2  || exit 1;
  fi
  if [ $multi_stage_train -le 2 ]; then
      cp $dir/stage2/den.fst $dir/stage3/.
      echo "$0:remove sibling network regularizer outputs "
      echo "and raname chain-objective for sibling to train "
      echo "with chain objective output for sibling network. \n"
      echo "Teacher-student objective can be added in future."
      nnet3-am-copy --edits-config=$dir/stage3/configs/edits.config \
        $dir/stage2/final.mdl $dir/stage3/0.mdl || exit 1;
      mkdir -p $dir/stage3/configs
      cp -r $dir/stage2/configs $dir/stage3/configs || exit 1;
      steps/nnet3/chain/train.py --stage 0 \
      --cmd "$decode_cmd" \
      --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
      --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
      --chain.xent-regularize $xent_regularize \
      --chain.leaky-hmm-coefficient 0.1 \
      --chain.l2-regularize 0.00005 \
      --chain.apply-deriv-weights false \
      --chain.lm-opts="--num-extra-lm-states=2000" \
      --egs.dir "$common_egs_dir" \
      --egs.stage $get_egs_stage \
      --egs.opts "--frames-overlap-per-eg 0" \
      --egs.chunk-width $frames_per_eg \
      --trainer.num-chunk-per-minibatch $minibatch_size \
      --trainer.frames-per-iter 1500000 \
      --trainer.num-epochs 1 \
      --trainer.optimization.num-jobs-initial $num_jobs_initial \
      --trainer.optimization.num-jobs-final $num_jobs_final \
      --trainer.optimization.initial-effective-lrate $sibling_initial_effective_lrate \
      --trainer.optimization.final-effective-lrate $sibling_final_effective_lrate \
      --trainer.max-param-change $max_param_change \
      --cleanup.remove-egs $remove_egs \
      --feat-dir data/${train_set}_hires \
      --tree-dir $treedir \
      --lat-dir exp/tri4_lats_nodup$suffix \
      --dir $dir/stage3  || exit 1;

  fi
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;
exit 0;
