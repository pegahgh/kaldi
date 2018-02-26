#!/bin/bash

# This script can be used for training multilingual setup using different
# languages (specifically babel languages) with no shared phones.
# It will generates separate egs directory for each dataset and combine them
# during training.
# In the new multilingual training setup, mini-batches of data corresponding to
# different languages are randomly sampled egs.scp file, which are generated
# based on probability distribution that reflects the relative
# frequency of the data from each language.

# For all languages, we share all the hidden layers and there is separate final
# layer per language.
# The bottleneck layer can be added to network structure.

# The script requires you to have baseline PLP features for all languages.
# It generates 40dim MFCC + pitch features for all languages.

# The global iVector extractor is trained using all languages and the iVector
# extracts for all languages.

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e

remove_egs=false
cmd=queue.pl
srand=0
stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=2
num_jobs_final=8
speed_perturb=true
use_pitch=false
use_ivector=true
megs_dir=
alidir=tri5_ali
suffix=
feat_suffix=_hires      # The feature suffix describing features used in
                        # multilingual training
                        # _hires_mfcc -> 40dim MFCC
                        # _hire_mfcc_pitch -> 40dim MFCC + pitch
                        # _hires_mfcc_pitch_bnf -> 40dim MFCC +pitch + BNF
# corpora
# language list used for multilingual training
# The map for lang-name to its abreviation can be find in
# local/prepare_flp_langconf.sh
# e.g lang_list=(101-cantonese 102-assamese 103-bengali)
#lang_list=(101-cantonese 102-assamese 103-bengali)
lang_list=(swbd wsj ami-sdm ami-ihm)
lang2weight="0.3,1.0,1.0,1.0"

# The language in this list decodes using Hybrid multilingual system.
# e.g. decode_lang_list=(101-cantonese)
#decode_lang_list=(102-assamese 103-bengali)

ivector_suffix=  # if ivector_suffix = _gb, the iVector extracted using global iVector extractor
                   # trained on pooled data from all languages.
                   # Otherwise, it uses iVector extracted using local iVector extractor.
bnf_dim=           # If non-empty, the bottleneck layer with this dimension is added at two layers before softmax.
use_flp=false      # If true, fullLP training data and configs used for training.
dir=exp/nnet3/multi_bnf

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

[ -f local.conf ] && . ./local.conf

num_langs=${#lang_list[@]}

echo "$0 $@"  # Print the command line for logging
if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

for lang_index in `seq 0 $[$num_langs-1]`; do
  for f in data/${lang_list[$lang_index]}/train/{feats.scp,text} exp/${lang_list[$lang_index]}/$alidir/ali.1.gz exp/${lang_list[$lang_index]}/$alidir/tree; do
    [ ! -f $f ] && echo "$0: no such file $f" #&& exit 1;
  done
done

if [ "$speed_perturb" == "true" ]; then
  suffix=${suffix}_sp
fi

if $use_pitch; then feat_suffix=${feat_suffix}_pitch ; fi
dir=${dir}${suffix}

for lang_index in `seq 0 $[$num_langs-1]`; do
  echo "$0: extract high resolution 40dim MFCC + pitch for speed-perturbed data "
  echo "and extract alignment."
  local/nnet3/run_common_langs.sh --stage $stage \
    --speed-perturb $speed_perturb ${lang_list[$lang_index]} || exit;
done

if $use_ivector; then
  mkdir -p data/multi
  mkdir -p exp/multi/nnet3
  global_extractor=exp/multi/nnet3
  multi_dir_data=data/multi/train${suffix}_hires
  echo "$0: combine training data using all langs for training global i-vector extractor."
  if [ ! -f $multi_dir_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Pooling training data in $multi_dir_data on" `date`
    echo ---------------------------------------------------------------------
    mkdir -p $multi_dir_data
    combine_lang_list=""
    for lang_index in `seq 0 $[$num_langs-1]`;do
      combine_lang_list="$combine_lang_list data/${lang_list[$lang_index]}/train${suffix}_hires"
    done
    utils/combine_data.sh data/multi/train${suffix}_hires $combine_lang_list
    utils/validate_data_dir.sh --no-feats data/multi/train${suffix}_hires
    touch data/multi/train${suffix}_hires/.done
  fi
  # create subset of multi data dir by taking the 1/8th of the data to be used
  # for the diag ubm training.
  multi_data_dir=data/multi/train${suffix}_1k_spk_hires
  utils/subset_data_dir.sh --speakers data/multi/train${suffix}_hires 1000 $multi_data_dir

  if [ ! -f $global_extractor/extractor/.done ]; then
    echo "$0: Generate global i-vector extractor on pooled data from all "
    echo "languages in $multi_data_dir, using an LDA+MLLT transform trained "
    echo "on ${lang_list[0]}."
    local/nnet3/run_shared_ivector_extractor.sh  \
      --suffix $suffix --use-flp $use_flp \
      --stage $stage ${lang_list[0]} \
      $multi_data_dir $global_extractor || exit 1;
    touch $global_extractor/extractor/.done
  fi
  echo "$0: Extracts ivector for all languages using $global_extractor/extractor."
  for lang_index in `seq 0 $[$num_langs-1]`; do
    local/nnet3/extract_ivector_lang.sh --stage $stage \
      --train-set train$suffix \
      --ivector-suffix $ivector_suffix \
      ${lang_list[$lang_index]} \
      $global_extractor/extractor || exit;
  done
fi
exit 1;
for lang_index in `seq 0 $[$num_langs-1]`; do
  multi_data_dirs[$lang_index]=data/${lang_list[$lang_index]}/train${suffix}${feat_suffix}
  multi_egs_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3/egs${ivector_suffix}
  multi_ali_dirs[$lang_index]=exp/${lang_list[$lang_index]}/${alidir}${suffix}
  multi_ivector_dirs[$lang_index]=exp/${lang_list[$lang_index]}/nnet3/ivectors_train${suffix}${ivector_suffix}
done

if $use_ivector; then
  ivector_dim=$(feat-to-dim scp:${multi_ivector_dirs[0]}/ivector_online.scp -) || exit 1;
else
  echo "$0: Not using iVectors in multilingual training."
  ivector_dim=0
fi
feat_dim=`feat-to-dim scp:${multi_data_dirs[0]}/feats.scp -`

if [ $stage -le 9 ]; then
  echo "$0: creating multilingual neural net configs using the xconfig parser";
  if [ -z $bnf_dim ]; then
    bnf_dim=1024
  fi
  input_layer_dim=$[3*$feat_dim+$ivector_dim]
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$ivector_dim name=ivector
  input dim=$feat_dim name=input
  output name=output-tmp input=Append(-1,0,1,ReplaceIndex(ivector, t, 0))

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(input@-2,input@-1,input,input@1,input@2,ReplaceIndex(ivector, t, 0)) dim=$input_layer_dim
  relu-renorm-layer name=tdnn2 dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn_bn dim=$bnf_dim
  # adding the layers for diffrent language's output
EOF
  # added separate outptut layer and softmax for all languages.
  for lang_index in `seq 0 $[$num_langs-1]`;do
    num_targets=`tree-info ${multi_ali_dirs[$lang_index]}/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;

    echo " relu-renorm-layer name=prefinal-affine-lang-${lang_index} input=tdnn_bn dim=1024"
    echo " output-layer name=output-${lang_index} dim=$num_targets max-change=1.5"
  done >> $dir/configs/network.xconfig

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ \
    --nnet-edits="rename-node old-name=output-0 new-name=output"

  cat <<EOF >> $dir/configs/vars
add_lda=false
include_log_softmax=false
EOF

  # removing the extra output node "output-tmp" added for back-compatiblity with
  # xconfig to config conversion.
  nnet3-copy --edits="remove-output-nodes name=output-tmp" $dir/configs/ref.raw $dir/configs/ref.raw || exit 1;
fi

if [ $stage -le 10 ]; then
  echo "$0: Generates separate egs dir per language for multilingual training."
  # sourcing the "vars" below sets
  #model_left_context=(something)
  #model_right_context=(something)
  #num_hidden_layers=(something)
  . $dir/configs/vars || exit 1;
  ivec="${multi_ivector_dirs[@]}"
  if $use_ivector; then
    ivector_opts=(--online-multi-ivector-dirs "$ivec")
  fi
  local/nnet3/prepare_multilingual_egs.sh --cmd "$decode_cmd" \
    "${ivector_opts[@]}" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --left-context $model_left_context --right-context $model_right_context \
    $num_langs ${multi_data_dirs[@]} ${multi_ali_dirs[@]} ${multi_egs_dirs[@]} || exit 1;

fi

if [ -z $megs_dir ];then
  megs_dir=$dir/egs
fi

if [ $stage -le 11 ] && [ ! -z $megs_dir ]; then
  echo "$0: Generate multilingual egs dir using "
  echo "separate egs dirs for multilingual training."
  common_egs_dir="${multi_egs_dirs[@]} $megs_dir"
  steps/nnet3/multilingual/combine_egs.sh $egs_opts \
    --cmd "$decode_cmd" \
    --samples-per-iter 400000 \
    $num_langs ${common_egs_dir[@]} || exit 1;
fi

if [ $stage -le 12 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir ${multi_data_dirs[0]} \
    --feat.online-ivector-dir ${multi_ivector_dirs[0]} \
    --egs.dir $megs_dir \
    --use-dense-targets false \
    --targets-scp ${multi_ali_dirs[0]} \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --use-gpu true \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 13 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
    lang_dir=$dir/${lang_list[$lang_index]}
    mkdir -p  $lang_dir
    echo "$0: rename output name for each lang to 'output' and "
    echo "add transition model."
    nnet3-copy --edits="rename-node old-name=output-$lang_index new-name=output" \
      $dir/final.raw - | \
      nnet3-am-init ${multi_ali_dirs[$lang_index]}/final.mdl - \
      $lang_dir/final.mdl || exit 1;
    cp $dir/cmvn_opts $lang_dir/cmvn_opts || exit 1;
    echo "$0: compute average posterior and readjust priors for language ${lang_list[$lang_index]}."
    steps/nnet3/adjust_priors.sh --cmd "$decode_cmd" \
      --use-gpu true \
      --iter final --use-raw-nnet false --use-gpu true \
      $lang_dir ${multi_egs_dirs[$lang_index]} || exit 1;
  done
fi

# decoding different languages
if [ $stage -le 14 ]; then
  num_decode_lang=${#decode_lang_list[@]}
  for lang_index in `seq 0 $[$num_decode_lang-1]`; do
    if [ ! -f $dir/${decode_lang_list[$lang_index]}/decode_dev10h.pem/.done ]; then
      echo "Decoding lang ${decode_lang_list[$lang_index]} using multilingual hybrid model $dir"
      run-4-anydecode-langs.sh --use-ivector $use_ivector \
        --nnet3-dir $dir --iter final_adj --use-flp $use_flp \
        ${decode_lang_list[$lang_index]} || exit 1;
      touch $dir/${decode_lang_list[$lang_index]}/decode_dev10h.pem/.done
    fi
  done
fi
