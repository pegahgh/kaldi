// cudamatrix/cu-math.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (Author: Daniel Povey)
//                2016  David Snyder

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



#ifndef KALDI_CUDAMATRIX_CU_MATH_H_
#define KALDI_CUDAMATRIX_CU_MATH_H_
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-device.h"
#include "base/timer.h"

namespace kaldi {

namespace cu {

/// RegularizeL1 is a gradient step with l1 regularization added to the
/// gradient.  We don't let the value cross over zero from positive to negative
/// or vice versa, in a single step.  If an element tries to cross zero and is
/// stopped, we zero the gradient.  (Dan: not sure why).
template<typename Real>
void RegularizeL1(CuMatrixBase<Real> *weight, CuMatrixBase<Real> *gradient,
                  Real l1_penalty, Real learning_rate);

/// Copies a permutation of src into tgt. The row permutation is specified in
/// copy_from_idx such that src.Row(copy_from_idx[r]) == tgt.Row(r). The
/// dimensions of copy_from_idx must be equivalent to the number of rows in
/// tgt and src and all elements in the vector must be in [0, src.numRows()-1].
template<typename Real>
void Randomize(const CuMatrixBase<Real> &src,
               const CuArray<int32> &copy_from_idx,
               CuMatrixBase<Real> *tgt);

/// Splice concatenates frames of src as specified in frame_offsets into tgt.
/// The dimensions of tgt must be equivalent to the number of rows in src
/// and it must be that tgt.NumColumns == src.NumColumns * frame_offsets.Dim().
/// As a result, tgt(i, k*n_cols + j) == src(i + frame_offsets[k], j) for the
/// general case where i in [0..src.NumRows()-1],
/// k in [0..frame_offsets.Dim()-1], j in [0..src.NumRows()-1]
/// and n_cols = src.NumColumns(). If i + frame_offsets[k] is greater than the
/// number of rows in src or less than 0 than the right side of the equation
/// is replaced by src(src.NumRows()-1, j) or src(0, j) respectively, to avoid
/// an index out of bounds.
template<typename Real>
void Splice(const CuMatrixBase<Real> &src,
            const CuArray<int32> &frame_offsets,
            CuMatrixBase<Real> *tgt);

/// Copies elements from src into tgt as given by copy_from_indices.
/// The matrices src and tgt must have the same dimensions and
/// the dimension of copy_from_indices must equal the number of columns
/// in the src matrix. As a result, tgt(i, j) == src(i, copy_from_indices[j]).
/// Also see CuMatrix::CopyCols(), which is more general.
template<typename Real>
void Copy(const CuMatrixBase<Real> &src,
          const CuArray<int32> &copy_from_indices,
          CuMatrixBase<Real> *tgt);

template <typename Real>
void Group2norm(const CuMatrixBase<Real> &src,
                CuMatrixBase<Real> *dest,
                int32 group_stride);

/*
  This function is used in computing the objective function and derivatives
  in xvector training.
  @param [in] scores   'scores' is a symmetric matrix of scores which are to
  be interpreted as log-odds (according to the model) of pairs coming from the
  same class, so scores(i, j) is the model's log p(same/different) for
  elements i and j of the original minibatch of input. We assume that the data
  in 'scores' has been arranged in such a way that pairs of indexes of the form
  (2k, 2k+1), e.g., (0, 1), (2, 3), (4, 5), etc, are from the same class, but
  indexes of any other form, such as (0, 2), (1, 2), etc, are from different
  classes.
  @param [out] objf_terms   'objf_terms' is a matrix of the same dimension as
  'scores' whose elements we will sum to get the objective function for this
  minibatch. This function computes the appropriate contributions to the
  objective function, as follows.
    if i == j:
      objf_terms(i, j)== 0       # the same exact element is not scored
    elsif i%2 == j%2:
      objf_terms(i, j) = log(p(same))
                       = -log(1 + exp(-scores(i, j))
    else:
      objf_terms(i, j) = 1 / (scores.NumRows() - 2) * log(p(different))
                       = -1/(scores.NumRows() - 2) * log(1+exp(scores(i,j))
  @param [out] objf_derivs    Element (i,j) of this matrix is the derivative
  of objf_terms(i,j) with respect to scores(i, j).
*/
void ComputeXvectorObjfFromScores(const CuMatrixBase<BaseFloat> &scores,
                                  CuMatrixBase<BaseFloat> *objf_terms,
                                  CuMatrixBase<BaseFloat> *objf_derivs,
                                  BaseFloat diss_penalty = 1.0);


} // namespace cu
} // namespace kaldi


#endif
