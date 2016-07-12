#!/bin/bash
# _v2 is as _v1 but has larger context and relu dim is 800 and CNT added to language list.
# This is a crosslingual training setup where there are no shared phones.
# It will generate separate egs directory fo each dataset and combine them 
# during training.
# For all languages, we share all the hidden layers but there are separate final
# layers. 
# The script requires you to have features for all languages. 

. cmd.sh 

stage=0
train_stage=-10
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=2
num_jobs_final=6
speed_perturb=true
use_pitch=true
use_entropy=false
use_ivector=true
use_sep_init_layer=false

alidir=tri5_ali
suffix=
aux_suffix=_hires_mfcc
# corpora
#lang_list=(ASM  BNG CNT LAO TAM  TGL  TUR  VTN  ZUL)
#lang_list=(ASM BNG  CNT  HAI  LAO  PSH  TAM  TGL  TUR  VTN  ZUL)
#lang_list=(ASM CNT BNG HAI  LAO  PSH  TAM  TGL  TUR  VTN  ZUL)
lang_list=(ASM CNT BNG HAI  LAO  PSH  TAM  TGL  TUR  VTN  ZUL)
#lang_list=(ASM BNG)
num_lang=${#lang_list[@]}
dir=exp/nnet3/multi_ASM_BNG_CNT_HAI_LAO_PSH_TAM_TGL_TUR_VTN_ZUL_v2
relu_dim=800
#splice_indexes="-2,-1,0,1,2 0 -2,2 0 -7,2 0"
splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 0"
frames_per_eg=8
avg_num_archives=12
cmd=$train_cmd
init_lrate=0.0017
final_lrate=0.00017
num_epochs=4

. cmd.sh
. ./path.sh

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh 

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi


for i in `seq 0 $[$num_lang-1]`; do
  for f in data/${lang_list[$i]}/train/{feats.scp,text} exp/${lang_list[$lang]}/$alidir/ali.1.gz exp/${lang_list[$lang]}/$alidir/tree; do
   [ ! -f $f ] && echo "$0: no such file $f" && exit 1; 
  done
done

if [ "$speed_perturb" == "true" ]; then
  suffix=${suffix}_sp
fi

ivector_suffix=
# if true, ivector extractor trained on pooled data from all languages.
if  ! $use_sep_init_layer && $use_ivector; then
  ivector_suffix=_gb
fi
echo use_ivector = $use_ivector and use_sep_init_layer = $use_sep_init_layer and ivector_suffix = $ivector_suffix

if $use_pitch; then aux_suffix=${aux_suffix}_pitch ; fi
if $use_entropy;then aux_suffix=${aux_suffix}_entropy ; fi  
dir=${dir}${suffix}

# extract high resolution MFCC features for speed-perturbed data
# and extract alignment 
for lang in `seq 0 $[$num_lang-1]`; do
  local/nnet3/run_common_langs.sh --stage $stage \
    --speed-perturb $speed_perturb ${lang_list[$lang]} || exit;
done

if [ ! $use_sep_init_layer ]; then
  # combine training data for all langs for training extractor
  echo suffix = $suffix
  if [ ! -f data/multi/train${suffix}_hires/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Pooling training data in data/multi${suffix}_hires on" `date`
    echo ---------------------------------------------------------------------
    mkdir -p data/multi
    mkdir -p data/multi/train${suffix}_hires
    for lang in `seq 0 $[$num_lang-1]`;do
      combine_lang_list="$combine_lang_list data/${lang_list[$lang]}/train${suffix}_hires"
    done
    utils/combine_data.sh data/multi/train${suffix}_hires $combine_lang_list
    utils/validate_data_dir.sh --no-feats data/multi/train${suffix}_hires
    touch data/multi/train${suffix}_hires/.done
  fi
  # If we do not use separate initial layer per language
  # then we use ivector extractor trained on pooled data from all languages
  # using an LDA+MLLT transform arbitrarily chonsed from single language.
  if [ ! -f exp/multi/nnet3/extractor/.done ]; then
    local/nnet3/run_shared_ivector_extractor.sh --stage $stage $lda_mllt_lang || exit 1; 
    touch exp/multi/nnet3/extractor/.done
  fi
fi

# extract ivector for all languages.
for lang in `seq 0 $[$num_lang-1]`; do
  local/nnet3/run_ivector_common_langs.sh --stage $stage \
    --use-sep-init-layer $use_sep_init_layer \
    --speed-perturb $speed_perturb ${lang_list[$lang]} || exit;
done

# set num_leaves for all languages
for lang in `seq 0 $[$num_lang-1]`; do
  num_leaves=`tree-info exp/${lang_list[$lang]}/$alidir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
  num_multiple_leaves="$num_multiple_leaves $num_leaves"

  multi_egs_dirs[$lang]=exp/${lang_list[$lang]}/nnet3/egs_v2${ivector_suffix}
  multi_ali_dirs[$lang]=exp/${lang_list[$lang]}/tri5_ali${suffix}

done

online_ivector_dir=
if $use_ivector; then
  online_ivector_dir=exp/${lang_list[0]}/nnet3/ivectors_train${suffix}${ivector_suffix}
fi
if [ -z "${online_ivector_dir}" ]; then
  ivector_dim=0
else
  ivector_dim=$(feat-to-dim scp:${online_ivector_dir}/ivector_online.scp -) || exit 1;
fi
feat_dim=`feat-to-dim scp:data/${lang_list[0]}/train${suffix}${aux_suffix}/feats.scp -`



if [ $stage -le 9 ]; then
  mkdir -p $dir/log
  echo "$0: creating neural net config for multilingual setups"
   # create the config files for nnet initialization
  $cmd $dir/log/make_config.log \
  python steps/nnet3/multi/make_tdnn_configs.py  \
    --separate-init-layer-per-output $use_sep_init_layer \
    --splice-indexes "$splice_indexes"  \
    --feat-dim $feat_dim \
    --ivector-dim $ivector_dim  \
    --relu-dim $relu_dim \
    --use-presoftmax-prior-scale false \
    --num-multiple-targets  "$num_multiple_leaves"  \
    --include-lda-layer false \
   $dir/configs || exit 1;
  # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
  # matrix.  This first config just does any initial splicing that we do;
  # we do this as it's a convenient way to get the stats for the 'lda-like'
  # transform.
  $cmd $dir/log/nnet_init.log \
    nnet3-init --srand=-2 $dir/configs/init.config $dir/init.raw || exit 1;
fi

. $dir/configs/vars || exit 1;

if [ $stage -le 10 ]; then
  for lang in `seq 0 $[$num_lang-1]`; do
    echo "$0: Generate egs for ${lang_list[$lang]}"
    egs_dir=${multi_egs_dirs[$lang]}
    ali_dir=${multi_ali_dirs[$lang]}
    data=data/${lang_list[$lang]}/train${suffix}${aux_suffix}
    num_frames=$(steps/nnet2/get_num_frames.sh $data)
    echo num_frames = $num_frames
    # sets samples_per_iter to have approximately 
    # same number of archives per language.
    samples_per_iter=$[$num_frames/($avg_num_archives*$frames_per_eg)]
    online_ivector_dir=
    if $use_ivector; then
      online_ivector_dir=exp/${lang_list[$lang]}/nnet3/ivectors_train${suffix}${ivector_suffix}
    fi
    if [ ! -z "$egs_dir" ]; then
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
        utils/create_split_dir.pl \
         /export/b0{3,4,5,6}/$USER/kaldi-data/egs/${lang_list[$lang]}-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
      fi

      extra_opts=()
      [ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
      [ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
      [ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
      extra_opts+=(--left-context $left_context)
      extra_opts+=(--right-context $right_context)
      echo "$0: calling get_egs.sh"
      steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
          --samples-per-iter $samples_per_iter --stage $get_egs_stage \
          --cmd "$cmd" $egs_opts \
          --frames-per-eg $frames_per_eg \
          $data $ali_dir $egs_dir || exit 1;
      
    fi
  done
fi

if [ $stage -le 11 ]; then
  echo "$0: training mutilingual model."
  steps/nnet3/multi/train_tdnn.sh --cmd "$train_cmd" \
  --use-ivector $use_ivector \
  --separate-init-layer-per-output $use_sep_init_layer \
  --num-epochs $num_epochs --cleanup false \
  --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
  --stage $train_stage \
  --initial-effective-lrate $init_lrate --final-effective-lrate $final_lrate \
  "${lang_list[@]}" "${multi_ali_dirs[@]}" "${multi_egs_dirs[@]}" \
  $dir || exit 1;

fi

decode_lang_list=(CNT ASM VTN TUR HAI LAO)
# decoding different languages
if [ $stage -le 12 ]; then
  num_decode_lang=${#decode_lang_list[@]}
  (
  for lang in `seq 0 $[$num_decode_lang-1]`; do
    if [ ! -f $dir/${decode_lang_list[$lang]}/decode_dev10h.pem/.done ]; then 
      cp $dir/cmvn_opts $dir/${decode_lang_list[$lang]}/.
      echo decoding lang ${decode_lang_list[$lang]} using multilingual model $dir
      run-4-anydecode-langs.sh --use-sep-init-layer $use_sep_init_layer \
        --use-ivector $use_ivector --nnet3-dir $dir ${decode_lang_list[$lang]} || exit 1;
      touch $dir/${decode_lang_list[$lang]}/decode_dev10h.pem/.done
    fi
  done
  wait
  )
fi
