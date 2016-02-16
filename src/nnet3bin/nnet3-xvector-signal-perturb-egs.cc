// nnet3bin/nnet3-xvector-signal-perturb-egs.cc

// Copyright 2016 Pegah Ghahremani

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
#include "feat/signal-distort.h"

namespace kaldi {
namespace nnet3 {

ApplyPerturbation(XvectorPerturbOptions opts,
                  const CuMatrix<BaseFloat> &input_egs,
                  CuMatrix<BaseFloat> *perturb_egs) {

  PerturbXvectorSignal perturb_xvector(opts);
  
  // Generate random shift samples to shift egs. 
  if (opts.max_shift != 0.0) {
    int32 max_shift_int = static_cast<int32>(opts_.max_shift * opts_.frame_dim);
    // shift each eg with different shifts. 
    for (int32 eg_num = 0; eg_num < input_egs.NumCols(); eg_num++) {
      int32 eg_dim = input_egs.NumCols() - opts_.frame_dim;
      shift = RandInt(0, max_shift_int); 
      perturb_egs->CopyRowFromVec(input_egs.Row(eg_num).Range(shift, eg_dim)); 
    }
  }
  if (opts.rand_distort) {
    // randomly generate an zero-phase FIR filter with no zeros. 
    perturb_xvector.ComputeAndApplyRandDistortion(input_egs,
                                  perturb_egs);
  }
  if (!opts.noise_egs.empty()) { 
    // select random block of noise egs and add to input_egs
    CuMatrix<BaseFloat noise_egs_mat;
    bool binary = false;
    noise_egs_mat.ReadKaldiObject(noise_egs_mat, binary);
    if (noise_egs_mat.NumRows() < input_egs.NumRows()) {
      // repeat the noise_egs_mat blocks to have same length block
      // and randomly perturb the rows.
    } else {
      // randomly select submatrix of it.
      std::vector rand_noise_ind(input_egs.NumRows());
      SubMatrix<BaseFloat> ran_noise_egs(noise_egs, rand_noise_ind);
      perturb_egs->AddMat(rand_noise_egs);
    }
  }
  // Perturb speed of signal eg, 
  if (opts.max_time_stretch != 0.0) 
    perturb_xvector.TimeStretch(opts_.max_time_stretch, perturb_egs);
   
  // If nagation is true, the sample values are randomly negated
  // with some probability.
  if (opts.negation) {
  
  }
}

} // end of namespace nnet3
} // end of namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;   
    typedef kaldi::int64 int64;


    const char *usage =
        "Corrupts  the examples supplied via input pipe with different type of distortions\n"
        "such as additive noise, negation, random time shifts or random distortion.\n"
        "Usage: nnet3-xvector-signal-perturb-egs [options...] <egs-especifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-xvector-signal-perturb-egs --noise-egs=noise.egs\n"
        "--max-shift=0.2 --max-speed-perturb=0.1 --negation=true\n"
        "ark:input.egs akr:distorted.egs\n";
    ParseOptions po(usage);

    XvectorPerturbOptions perturb_opts;
    po.Read(perturb_opts);

    po.Read(argc, argv);
    if (po.NumArgc() != 2) {
      po.PrintUsage();
      exit(1):
    }
    
    std::string examples_rspecifier = po.GetArg(1),
      examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    
    NnetExampleWriter examle_writer(examples_wspecifier);

    int64 num_read = 0, num_written = 0;
    
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      std::string key = example_reader.Key();
      const NnetExample &eg = example_reader.Value();
      NnetExample *perturb_eg = new NnetExample();
      ApplyPerturbation(perturb_opts, eg, perturb_eg);
      examle_writer->Write(key, perturb_eg);
      num_written++;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
