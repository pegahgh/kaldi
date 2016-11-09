// cudamatrix/cu-math-test.cc

// Copyright 2013 Johns Hopkins University (Author: David Snyder)

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


#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-array.h"

#if defined(_MSC_VER)
#include <time.h>
#endif

using namespace kaldi;


namespace kaldi {


/*
 * Unit tests
 */

template<typename Real>
static void UnitTestCuMathRandomize() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  CuMatrix<Real> src(M, N);
  CuMatrix<Real> tgt(M, N);
  CuArray<int32> copy_from_idx;

  src.SetRandn();
  int32 n_rows = src.NumRows();
  int32 n_columns = src.NumCols();
  std::vector<int32> copy_from_idx_vec;

  for (int32 i = 0; i < n_rows; i++) {
    copy_from_idx_vec.push_back(Rand() % n_rows);
  }
  copy_from_idx.CopyFromVec(copy_from_idx_vec);
  cu::Randomize(src, copy_from_idx, &tgt);

  for (int32 i = 0; i < n_rows; i++) {
    for (int32 j = 0; j < n_columns; j++) {
      Real src_val = src(copy_from_idx_vec.at(i), j);
      Real tgt_val = tgt(i, j);
      AssertEqual(src_val, tgt_val);
    }
  }
}


template<typename Real>
static void UnitTestCuMathCopy() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  CuMatrix<Real> src(M, N);
  CuMatrix<Real> tgt(M, N);
  CuArray<int32> copy_from_idx;

  src.SetRandn();
  int32 n_rows = src.NumRows();
  int32 n_columns = src.NumCols();
  std::vector<int32> copy_from_idx_vec;

  for (int32 i = 0; i < n_columns; i++) {
    copy_from_idx_vec.push_back(Rand() % n_columns);
  }
  copy_from_idx.CopyFromVec(copy_from_idx_vec);
  cu::Copy(src, copy_from_idx, &tgt);

  for (int32 i = 0; i < n_rows; i++) {
    for (int32 j = 0; j < n_columns; j++) {
      Real src_val = src(i, copy_from_idx_vec.at(j));
      Real tgt_val = tgt(i, j);
      AssertEqual(src_val, tgt_val);
    }
  }
}

template<typename Real>
static void UnitTestCuMathSplice() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  CuMatrix<Real> src(M, N);
  CuArray<int32> frame_offsets;

  src.SetRandn();
  int32 n_rows = src.NumRows();
  int32 n_columns = src.NumCols();
  std::vector<int32> frame_offsets_vec;

  // The number of columns of tgt is rows(src)
  // times n_frame_offsets, so we keep n_frame_offsets
  // reasonably small (2 <= n <= 6).
  int32 n_frame_offsets = Rand() % 7 + 2;
  for (int32 i = 0; i < n_frame_offsets; i++) {
    frame_offsets_vec.push_back(Rand() % 2 * n_columns - n_columns);
  }

  CuMatrix<Real> tgt(M, N * n_frame_offsets);
  frame_offsets.CopyFromVec(frame_offsets_vec);
  cu::Splice(src, frame_offsets, &tgt);

  Matrix<Real> src_copy(src), tgt_copy(tgt);
  for (int32 i = 0; i < n_rows; i++) {
    for (int32 k = 0; k < n_frame_offsets; k++) {
      for (int32 j = 0; j < n_columns; j++) {
        Real src_val;
        if (i + frame_offsets_vec.at(k) >= n_rows) {
          src_val = src_copy(n_rows-1, j);
        } else if (i + frame_offsets_vec.at(k) <= 0) {
          src_val = src_copy(0, j);
        } else {
          src_val = src_copy(i + frame_offsets_vec.at(k), j);
        }
        Real tgt_val = tgt_copy(i, k * n_columns + j);
        AssertEqual(src_val, tgt_val);
      }
    }
  }
}

template<typename Real> void CudaMathUnitTest() {
  #if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported())
  #endif
  UnitTestCuMathRandomize<Real>();
  UnitTestCuMathSplice<Real>();
  UnitTestCuMathCopy<Real>();
}

static BaseFloat Difference(const CuMatrix<BaseFloat> &mat1,
                            const CuMatrix<BaseFloat> &mat2) {
  CuMatrix<BaseFloat> diff(mat1);
  diff.AddMat(-1.0, mat2);
  BaseFloat ans = diff.FrobeniusNorm() /
      (0.5 * mat1.FrobeniusNorm() + 0.5 * mat2.FrobeniusNorm());
  return ans;
}

static void UnitTestComputeXvectorObjfFromScores() {
  for (int32 i = 0; i < 10; i++) {
    int32 dim = RandInt(10, 30);
    CuMatrix<BaseFloat> scores(dim, dim),
        small_change(dim, dim);
    scores.SetRandn();
    small_change.SetRandn();
    // ensure it works for large scores as well as small.
    BaseFloat scale = (RandInt(0, 1) == 0 ? 1.0 : 20.0),
        delta = 0.01;
    scores.Scale(scale);
    CuMatrix<BaseFloat> objf_terms(dim, dim, kUndefined),
        objf_deriv(dim, dim, kUndefined);
    cu::ComputeXvectorObjfFromScores(scores, &objf_terms, &objf_deriv);

    scores.AddMat(delta, small_change);
    CuMatrix<BaseFloat> objf_terms2(dim, dim, kUndefined),
        objf_deriv2(dim, dim, kUndefined);
    cu::ComputeXvectorObjfFromScores(scores, &objf_terms2, &objf_deriv2);
    CuMatrix<BaseFloat> objf_terms2_predicted(objf_terms);
    objf_terms2_predicted.AddMatMatElements(delta, objf_deriv, small_change, 1.0);

    BaseFloat base_diff = Difference(objf_terms, objf_terms2),
        corrected_diff = Difference(objf_terms2, objf_terms2_predicted);
    KALDI_LOG << "Base diff is " << base_diff << ", corrected diff = "
              << corrected_diff;
    KALDI_ASSERT(corrected_diff < 0.1 * base_diff);
  }
}


} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("yes"); // -2 .. automatic selection
#endif
    srand(time(NULL));
    kaldi::CudaMathUnitTest<float>();

#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaMathUnitTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaMathUnitTest<float>();
#endif
    UnitTestComputeXvectorObjfFromScores();
    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

