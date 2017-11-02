#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# SRE16 trials
sre16_trials=data/sre16_eval_test/trials
sre16_trials_tgl=data/sre16_eval_test/trials_tgl
sre16_trials_yue=data/sre16_eval_test/trials_yue
nnet_dir=exp/xvector_age_nnet_1e_aug2
data=data/sre_08_10
fold_index=1
stage=1
train_stage=-10
# random label shift config
num_delays=3 # number of copies of data with random label shift, augmented to
             # original data.
label_delay=0.05 # age label are shifted by label_delay % of original age
                # to left or right by 50%
regression_regularize=0.0
. parse_options.sh || exit 1;

if [ $stage -le 0 ] && false; then
  # Path to some, but not all of the training corpora
  data_root=/export/corpora/LDC

  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $data_root/LDC2013S03 data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh $data_root data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1 data/sre2006_test_2 \
    data/sre08 data/mx6 data/sre10
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare NIST SRE 2016 evaluation data.
  local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data

  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
fi

if [ $stage -le 1 ]; then
  # Make filterbanks and compute the energy-based VAD for each dataset
  #for name in sre swbd sre16_eval_enroll sre16_eval_test sre16_major; do
  for name in sre_08_10; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  #utils/combine_data.sh data/swbd_sre data/swbd data/sre
  #utils/fix_data_dir.sh data/swbd_sre
fi

# In this section, we augment the data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  if false; then #100
  utils/data/get_utt2num_frames.sh --nj 40 --cmd "$train_cmd" $data
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data/utt2num_frames > $data/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    $data ${data}_reverb || exit 1;
  cp ${data}/vad.scp ${data}_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" ${data}_reverb ${data}_reverb.new
  rm -rf ${data}_reverb
  mv ${data}_reverb.new ${data}_reverb
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /export/corpora/JHU/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" ${data} ${data}_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" ${data} ${data}_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" ${data} ${data}_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh ${data}_aug ${data}_reverb ${data}_noise ${data}_music ${data}_babble
  fi #100
  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    ${data}_aug exp/make_mfcc $mfccdir

  # Combine the clean and augmented SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh ${data}_combined ${data}_aug ${data}
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 100 --cmd "$train_cmd" \
    ${data}_combined ${data}_combined_no_sil ${data}_combined_no_sil
  utils/fix_data_dir.sh ${data}_combined_no_sil
  utils/data/get_utt2num_frames.sh --nj 100 --cmd "$train_cmd" ${data}_combined_no_sil
  utils/fix_data_dir.sh ${data}_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=30
  mv ${data}_combined_no_sil/utt2num_frames ${data}_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' ${data}_combined_no_sil/utt2num_frames.bak > ${data}_combined_no_sil/utt2num_frames
  utils/filter_scp.pl ${data}_combined_no_sil/utt2num_frames ${data}_combined_no_sil/utt2spk > ${data}_combined_no_sil/utt2spk.new
  mv ${data}_combined_no_sil/utt2spk.new ${data}_combined_no_sil/utt2spk
  utils/fix_data_dir.sh ${data}_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=1
  awk '{print $1, NF-1}' ${data}_combined_no_sil/spk2utt > ${data}_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' ${data}_combined_no_sil/spk2num | utils/filter_scp.pl - ${data}_combined_no_sil/spk2utt > ${data}_combined_no_sil/spk2utt.new
  mv ${data}_combined_no_sil/spk2utt.new ${data}_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl ${data}_combined_no_sil/spk2utt > ${data}_combined_no_sil/utt2spk

  utils/filter_scp.pl ${data}_combined_no_sil/utt2spk ${data}_combined_no_sil/utt2num_frames > ${data}_combined_no_sil/utt2num_frames.new
  mv ${data}_combined_no_sil/utt2num_frames.new ${data}_combined_no_sil/utt2num_frames

  # Now we're reaady to create training examples.
  utils/validate_data_dir.sh --no-text ${data}_combined_no_sil
  utils/fix_data_dir.sh ${data}_combined_no_sil
fi

if [ $num_delays -gt 0 ] && [ $stage -le 7 ]; then
  echo "$0: Augment data dir with delayed version of age labels with "
  echo "increasing or decreasing age with $label_delay % of original age."
  dir_to_combine="${data}_combined_no_sil"
  for d in `seq $num_delays`;do
    delay_suffix="_ld_${d}"
    utils/copy_data_dir.sh --spk-prefix ld${d}- --utt-prefix ld${d}- ${data}_combined_no_sil ${data}_combined_no_sil${delay_suffix}
    RANDOM=$[$d+123]
    cat ${data}_combined_no_sil${delay_suffix}/utt2age | awk -v delay=$label_delay -v seed=$RANDOM '{srand(seed); print $1,$2+(2*int(2*rand())-1)*int(1.0+$2*delay*rand())}' > ${data}_combined_no_sil${delay_suffix}/utt2age.bkup
    mv ${data}_combined_no_sil${delay_suffix}/utt2age.bkup ${data}_combined_no_sil${delay_suffix}/utt2age
    utils/fix_data_dir.sh ${data}_combined_no_sil${delay_suffix}
    dir_to_combine="$dir_to_combine ${data}_combined_no_sil${delay_suffix}"
  done

  delay_suffix="_ld${label_delay}_${num_delays}"
  utils/combine_data.sh ${data}_combined_no_sil${delay_suffix} $dir_to_combine

  utils/data/get_utt2num_frames.sh --nj 100 --cmd "$train_cmd" ${data}_combined_no_sil${delay_suffix}

  utils/fix_data_dir.sh ${data}_combined_no_sil${delay_suffix}

  rm -rf ${data}_combined_no_sil_ld_*
fi
delay_suffix="_ld${label_delay}_${num_delays}"
test_data=$data/${fold_index}/test
train_data=${data}_combined_no_sil${delay_suffix}
if [ $stage -le 7 ]; then
# exclude test data from training dataset and use new dataset for training
# all different version of utt from test set are removed from training data.
  local/prepare_data_age.sh $fold_index ${data} ${data}/${fold_index}
  for utt in `cat $test_data/utt2spk | awk '{print $1}'`; do
    grep $utt $train_data/utt2spk | awk '{print $1}';
  done > $test_data/utt_to_exclude
  rm -rf $train_data/${fold_index}
  mkdir -p $train_data/${fold_index}
  awk '{print $1}' $train_data/utt2spk | utils/filter_scp.pl --exclude $test_data/utt_to_exclude > $train_data/${fold_index}/train_utt_list
  utils/subset_data_dir.sh --utt-list $train_data/${fold_index}/train_utt_list \
    $train_data ${train_data}/${fold_index}/train
fi

if [ $stage -le 8 ]; then
  local/nnet3/xvector/run_xvector_age.sh --stage $train_stage --train-stage -1 \
    --num-repeats 100 --frames-per-iter 1600000000 --num-epochs 10 \
    --regression-regularize 10 \
    --frames-per-iter-diagnostic 100000000 \
    --data ${train_data}/${fold_index}/train --nnet-dir $nnet_dir/${fold_index} \
    --egs-dir $nnet_dir/${fold_index}/egs
    --regression-regularize $regression_regularize
fi

if [ $stage -le 9 ]; then
  # The SRE16 test data for test.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    $nnet_dir/${fold_index} data/sre_08_10/${fold_index}/test \
    $nnet_dir/${fold_index}/xvectors_test

  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    --extract-config extract.config.regression \
    $nnet_dir/${fold_index} data/sre_08_10/${fold_index}/test \
    $nnet_dir/${fold_index}/xvectors_test_regression
fi
cp $nnet_dir/${fold_index}/age_offset $data/${fold_index}/test/

if [ $stage -le 10 ]; then
  echo "Evaluate trained model on fold ${fold_index}."
  local/evaluate_age.sh $data/${fold_index}/test \
    $nnet_dir/${fold_index}/xvectors_test || exit 1;
  local/evaluate_age.sh $data/${fold_index}/test \
    $nnet_dir/${fold_index}/xvectors_test_regression || exit 1;
fi

if [ $stage -le 11 ]; then
  logistic_dir=$nnet_dir/${fold_index}/logistic_regression
  logistic_mdl=$logistic_dir/final.mdl
  age2int=$nnet_dir/${fold_index}/age2int
  if [ ! -f $age2int ]; then echo "$age2int does not exists." && exit 1; fi
  mkdir -p ${logistic_dir}
  mkdir -p ${logistic_dir}/train
  mkdir -p ${logistic_dir}/test
  echo "train logistic regression on top of extracted embedding."
  echo "Extract embedding for original training data."
  echo "output-node name=output input=tdnn6.affine" > $nnet_dir/${fold_index}/extract.config.embedding

  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 100 \
    --extract-config extract.config.embedding \
    $nnet_dir/${fold_index} data/sre_08_10/${fold_index}/train \
    $nnet_dir/${fold_index}/xvectors_train_embedding

  echo "Extract embeddding for test data."
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    --extract-config extract.config.embedding \
    $nnet_dir/${fold_index} data/sre_08_10/${fold_index}/test \
    $nnet_dir/${fold_index}/xvectors_test_embedding

  echo "Train logistic regeression on top of embedding."
    logistic-regression-train --config=conf/logistic-regression.conf \
      scp:$nnet_dir/${fold_index}/xvectors_train_embedding/xvector.scp \
      "ark:utils/apply_map.pl -f 2 $age2int < data/sre_08_10/${fold_index}/train/utt2age |" \
      $logistic_mdl > $nnet_dir/${fold_index}/logistic_regression/logistic_regression.log

  echo "$0: find the prior scale vector based on data distribution on test data."
  echo "$0: Evaluate model on training data."
  logistic-regression-eval --apply-log=true ${logistic_mdl} \
    scp:$nnet_dir/${fold_index}/xvectors_train_embedding/xvector.scp \
    ark,t:${logistic_dir}/train/posteriors

  cat ${logistic_dir}/train/posteriors | \
    awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max)
                            { max=$f; argmax=f; }}
                            print $1, (argmax - 3); }' > ${logistic_dir}/train/output
  compute-wer --mode=present --text \
    "ark:apply_map.pl -f 2 $age2int < data/sre_08_10/${fold_index}/train/utt2age |"\
    ark:${logistic_dir}/train/output \
    > ${logistic_dir}/train/wer

  utils/apply_map.pl -f 2 $age2int < data/sre_08_10/${fold_index}/train/utt2age \
    > ${logistic_dir}/train/utt2age.mapped

  echo "$0: Evaluate model on test data."
  logistic-regression-eval --apply-log=true ${logistic_mdl} \
    scp:$nnet_dir/${fold_index}/xvectors_test_embedding/xvector.scp \
    ark,t:${logistic_dir}/test/posteriors

  cat ${logistic_dir}/test/posteriors | \
    awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max)
                            { max=$f; argmax=f; }}
                            print $1, (argmax - 3); }' > ${logistic_dir}/test/output

  utils/apply_map.pl -f 2 $age2int < data/sre_08_10/${fold_index}/test/utt2age \
    > ${logistic_dir}/test/utt2age.mapped

  compute-wer --mode=present --text \
    "ark:utils/apply_map.pl -f 2 $age2int < data/sre_08_10/${fold_index}/test/utt2age |" \
    ark:${logistic_dir}/test/output \
    > ${logistic_dir}/test/wer

  mean_abs_error=`paste <(awk '{print $2}' ${logistic_dir}/test/output) \
    <(utils/apply_map.pl -f 2 $age2int < data/sre_08_10/1/test/utt2age | awk '{print $2}') | \
    awk '{ sum +=( $1-$2 >= 0 ? $1-$2 : $2-$1 ); n++ } END { if (n > 0) print sum / n; }'`
  echo "mean abs error |y_real - y_predicted| is $mean_abs_error." > ${logistic_dir}/test/mean_abs_error
fi
