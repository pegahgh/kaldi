// bin/matrix-sum-rows.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Sum the rows of an input table of matrices and output the corresponding\n"
        "table of vectors\n"
        "\n"
        "Usage: matrix-sum-rows [options] <matrix-rspecifier> <vector-wspecifier>\n"
        "e.g.: matrix-sum-rows ark:- ark:- | vector-sum ark:- sum.vec\n"
        "See also: matrix-sum, vector-sum\n";

    bool average = false;
    int32 num_rows = 0;
    ParseOptions po(usage);

    po.Register("average", &average, "If true, compute average instead of "
                "sum.");
    po.Register("n", &num_rows, "If non_zero, the output is sum of n "
                "consequent rows of matrix (average, if average is true)."
                "If n < 0, summation contains the last n rows and n > 0, then "
                "the first n rows are summed.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);
    
    SequentialBaseFloatMatrixReader mat_reader(rspecifier);
    BaseFloatVectorWriter vec_writer(wspecifier);
    
    int32 num_done = 0;
    int64 num_rows_done = 0;
    for (; !mat_reader.Done(); mat_reader.Next()) {
      std::string key = mat_reader.Key();
      Matrix<double> mat(mat_reader.Value());
      Vector<double> vec(mat.NumCols());
      int32 r1 = 0, r2 = mat.NumRows();
      if (num_rows < -1 * r2 || num_rows > r2) {
        KALDI_LOG << "Num of frames, " << r2
                  << ", is less than --n=" << abs(num_rows);
      } else {
        if (num_rows != 0) {
            r1 = (num_rows > 0 ? 0 : r2 + num_rows);
            r2 = (num_rows > 0 ? 1 : -1) * num_rows;
        }
      }
      vec.AddRowSumMat(1.0, mat.RowRange(r1,r2), 0.0);
      // Do the summation in double, to minimize roundoff.
      Vector<BaseFloat> float_vec(vec);
      if (average)
        float_vec.Scale(1.0/float(r2));
      vec_writer.Write(key, float_vec);
      num_done++;
      num_rows_done += mat.NumRows();
    }
    
    KALDI_LOG << "Summed rows " << num_done << " matrices, "
              << num_rows_done << " rows in total.";
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


