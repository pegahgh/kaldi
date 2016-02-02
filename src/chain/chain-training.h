// chain/chain-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_CHAIN_CHAIN_TRAINING_H_
#define KALDI_CHAIN_CHAIN_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {


struct ChainTrainingOptions {
  // l2 regularization constant on the 'chain' output; the actual term added to
  // the objf will be -0.5 times this constant times the squared l2 norm.
  // (squared so it's additive across the dimensions).  e.g. try 0.0005.
  BaseFloat l2_regularize;

  // Coefficient for 'leaky hmm'.  This means we have an epsilon-transition from
  // each state to a special state with probability one, and then another
  // epsilon-transition from that special state to each state, with probability
  // leaky_hmm_coefficient times [initial-prob of destination state].  Imagine
  // we make two copies of each state prior to doing this, version A and version
  // B, with transition from A to B, so we don't have to consider epsilon loops-
  // or just imagine the coefficient is small enough that we can ignore the
  // epsilon loops.
  BaseFloat leaky_hmm_coefficient;


  // Cross-entropy regularization constant.  (e.g. try 0.1).  If nonzero,
  // the network is expected to have an output named 'output-xent', which
  // should have a softmax as its final nonlinearity.
  BaseFloat xent_regularize;

  ChainTrainingOptions(): l2_regularize(0.0), leaky_hmm_coefficient(1.0e-05),
                          xent_regularize(0.0) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("l2-regularize", &l2_regularize, "l2 regularization "
                   "constant for 'chain' training, applied to the output "
                   "of the neural net.");
    opts->Register("leaky-hmm-coefficient", &leaky_hmm_coefficient, "Coefficient "
                   "that allows transitions from each HMM state to each other "
                   "HMM state, to ensure gradual forgetting of context (can "
                   "improve generalization).  For numerical reasons, may not be "
                   "exactly zero.");
    opts->Register("xent-regularize", &xent_regularize, "Cross-entropy "
                   "regularization constant for 'chain' training.  If "
                   "nonzero, the network is expected to have an output "
                   "named 'output-xent', which should have a softmax as "
                   "its final nonlinearity.");
  }
};


/**
   This function does both the numerator and denominator parts of the 'chain'
   computation in one call.

   @param [in] opts        Struct containing options
   @param [in] den_graph   The denominator graph, derived from denominator fst.
   @param [in] supervision  The supervision object, containing the supervision
                            paths and constraints on the alignment as an FST
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).  The rows are ordered as: all sequences
                            for frame 0; all sequences for frame 1; etc.
   @param [in] xent_output  The output of cross entropy; dimension mus equal nnet_output
                            If non-NULL, the l2 regularization regresses 
                            nnet_output to be linear function of xent_output.
   @param [out] objf       The [num - den] objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] l2_term  The l2 regularization term in the objective function, if
                           the --l2-regularize option is used.  To be added to 'o
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.
   @param [out] nnet_output_deriv  The derivative of the objective function w.r.t.
                           the neural-net output.  Only written to if non-NULL.
                           You don't have to zero this before passing to this function,
                           we zero it internally.
   @param [out] xent_output_deriv  If non-NULL, then the numerator part of the derivative
                           (which equals a posterior from the numerator forward-backward,
                           scaled by the supervision weight) is written to here.  This will
                           be used in the cross-entropy regularization code.  This value
                           is also used in computing the cross-entropy objective value.
     
*/
void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv,
                              CuMatrixBase<BaseFloat> *xent_output_deriv = NULL);
                              
/**
   This function computes offset and scale, where Y is regressed to
   be linear function of X. Linear least square method is
   used to compute scale and offset coefficients.

   The objecitve is to minimize L w.r.t scale_i, offset_i, 
   L = \sum_{j=1}^{n}(\sum_i (y_ji - target_ji)^2),
   where the target_ji = scale_i * x_ji + offset_i.

   scale_i = [\sum_j (x_ji * y_ji) - 
             1/n * \sum_j(x_ji) * \sum_j(y_ji)] / 
             [\sum_j(x_ji^2) - 1/n * (\sum_j(x_ji))^2]
   offset_i = 1/n * \sum_j (y_ji - scale_i * x_ji)
   where n is the number of exampls.

   @param [in] x        The elements of x contains examples for independent variable X.
   @param [in] y        The elements of y contains examples for dependent variable Y.

   @param [out] scale   The scale parameter  
   @param [out] offset  The offset parameter 
 **/
void LeastSquareRegression(const CuMatrixBase<BaseFloat> &x,
                           const CuMatrixBase<BaseFloat> &y,
                           CuVector<BaseFloat> *scale,
                           CuVector<BaseFloat> *offset); 

/**
  This function computes weighted l2-regularization term and updates the 
  derivatives. The weighted l2-regularization is a weighted combination of output dims,
  and the objective is L = \sum_{j=1}^{n}(\sum_i weight_i * (y_ji - target_ji)^2).
  If xent_output is not null, l2_term regress 
  output of chain output to be linear function of cross-entropy output,
  and target_ji = alpha_i * x_ji + offset_i.
  Otherwise, the l2_term minimize l2_norm of chain output and target_ji = 0.
**/
void ComputeRegularizationTerm(const CuMatrixBase<BaseFloat> &nnet_output,
                               const CuMatrixBase<BaseFloat> *xent_output,
                               const CuVectorBase<BaseFloat> *weight,
                               BaseFloat l2_regularize,
                               BaseFloat *l2_term,
                               CuMatrixBase<BaseFloat> *nnet_output_deriv,
                               CuMatrixBase<BaseFloat> *xent_output_deriv = NULL);

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_TRAINING_H_

