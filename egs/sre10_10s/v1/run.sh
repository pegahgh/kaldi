#!/bin/bash
# Copyright 2015-2016   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
trials=data/sre10_test_10s/trials
trials_female=data/sre10_test_10s_female/trials
trials_male=data/sre10_test_10s_male/trials
num_components=2048 # Larger than this doesn't make much of a difference.

# Prepare the 10s-10s portion of the SRE 2010 evaluation data.
local/make_sre_2010_test_10s.pl /export/corpora5/SRE/SRE2010/eval/ data/
local/make_sre_2010_train_10s.pl /export/corpora5/SRE/SRE2010/eval/ data/

# Prepare a collection of NIST SRE data prior to 2010. This is
# used to train the PLDA model and is also combined with SWB
# for UBM and i-vector extractor training data.
local/make_sre.sh data

# Prepare SWB for UBM and i-vector extractor training.
local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
  data/swbd2_phase2_train
local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
  data/swbd2_phase3_train
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
  data/swbd_cellular1_train
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
  data/swbd_cellular2_train

utils/combine_data.sh data/train \
  data/swbd_cellular1_train data/swbd_cellular2_train \
  data/swbd2_phase2_train data/swbd2_phase3_train data/sre

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_train_10s exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_test_10s exp/make_mfcc $mfccdir

for name in sre sre10_train_10s sre10_test_10s train; do
  utils/fix_data_dir.sh data/${name}
done

sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_train_10s exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_test_10s exp/make_vad $vaddir

for name in sre sre10_train_10s sre10_test_10s train; do
  utils/fix_data_dir.sh data/${name}
done

# Reduce the amount of training data for the UBM.
utils/subset_data_dir.sh data/train 16000 data/train_16k
utils/subset_data_dir.sh data/train 32000 data/train_32k

# Train UBM and i-vector extractor.
sid/train_diag_ubm.sh --cmd "$train_cmd -l mem_free=20G,ram_free=20G" \
  --nj 20 --num-threads 8 \
  data/train_16k $num_components \
  exp/diag_ubm_$num_components

sid/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd -l mem_free=25G,ram_free=25G" data/train_32k \
  exp/diag_ubm_$num_components exp/full_ubm_$num_components

sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
  --ivector-dim 600 \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/train \
  exp/extractor

# Extract i-vectors.
sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor data/sre10_train_10s \
  exp/ivectors_sre10_train_10s

sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor data/sre10_test_10s \
  exp/ivectors_sre10_test_10s

sid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
  exp/extractor data/sre \
  exp/ivectors_sre

# Separate the i-vectors into male and female partitions and calculate
# i-vector means used by the scoring scripts.
local/scoring_common.sh data/sre data/sre10_train_10s data/sre10_test_10s \
  exp/ivectors_sre exp/ivectors_sre10_train_10s exp/ivectors_sre10_test_10s

# The commented out scripts show how to do cosine scoring with and without
# first reducing the i-vector dimensionality with LDA. PLDA tends to work
# best, so we don't focus on the scores obtained here.
#
# local/cosine_scoring.sh data/sre10_train data/sre10_test \
#  exp/ivectors_sre10_train exp/ivectors_sre10_test $trials local/scores_gmm_2048_ind_pooled
# local/lda_scoring.sh data/sre data/sre10_train data/sre10_test \
#  exp/ivectors_sre exp/ivectors_sre10_train exp/ivectors_sre10_test $trials local/scores_gmm_2048_ind_pooled

# Create a gender independent PLDA model and do scoring.
local/plda_scoring.sh data/sre data/sre10_train_10s data/sre10_test_10s \
  exp/ivectors_sre exp/ivectors_sre10_train_10s exp/ivectors_sre10_test_10s $trials local/scores_gmm_2048_ind_pooled
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_10s_female data/sre10_test_10s_female \
  exp/ivectors_sre exp/ivectors_sre10_train_10s_female exp/ivectors_sre10_test_10s_female $trials_female local/scores_gmm_2048_ind_female
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_10s_male data/sre10_test_10s_male \
  exp/ivectors_sre exp/ivectors_sre10_train_10s_male exp/ivectors_sre10_test_10s_male $trials_male local/scores_gmm_2048_ind_male

# Create gender dependent PLDA models and do scoring.
local/plda_scoring.sh data/sre_female data/sre10_train_10s_female data/sre10_test_10s_female \
  exp/ivectors_sre exp/ivectors_sre10_train_10s_female exp/ivectors_sre10_test_10s_female $trials_female local/scores_gmm_2048_dep_female
local/plda_scoring.sh data/sre_male data/sre10_train_10s_male data/sre10_test_10s_male \
  exp/ivectors_sre exp/ivectors_sre10_train_10s_male exp/ivectors_sre10_test_10s_male $trials_male local/scores_gmm_2048_dep_male

mkdir -p local/scores_gmm_2048_dep_pooled
cat local/scores_gmm_2048_dep_male/plda_scores local/scores_gmm_2048_dep_female/plda_scores \
  > local/scores_gmm_2048_dep_pooled/plda_scores


# TODO baseline results without training a PLDA model split into 10s chunks.
# This will need to be done at some point.

# GMM-2048 EER
# ind female: 12.68
# ind male: 10.31
# ind pooled: 11.54
# dep female: 13.03
# dep male: 11.07
# dep pooled: 12.09
echo "GMM-$num_components EER"
for x in ind dep; do
  for y in female male pooled; do
    eer=`compute-eer <(python local/prepare_for_eer.py $trials local/scores_gmm_${num_components}_${x}_${y}/plda_scores) 2> /dev/null`
    echo "${x} ${y}: $eer"
  done
done
