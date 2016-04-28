// nnet3bin/nnet3-copy.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)
//           2015  Xingyu Na

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

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
void RenameNodes(const std::string nodes_to_rename, nnet3::Nnet *nnet) {
  std::vector<std::string> name_pairs;
  SplitStringToVector(nodes_to_rename, ",", false, &name_pairs);
  for (int32 i = 0; i < name_pairs.size(); i++) {
    std::vector<std::string> name_pair;
    SplitStringToVector(name_pairs[i], ":", false, &name_pair);
    if (name_pair.size() != 2)
      KALDI_ERR << "Malformed argument to option --rename-output-nodes";
    nnet->RenameOutputNode(name_pair[0], name_pair[1]);
  }
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy 'raw' nnet3 neural network to standard output\n"
        "Also supports setting all the learning rates to a value\n"
        "(the --learning-rate option)\n"
        "\n"
        "Usage:  nnet3-copy [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet3-copy --binary=false 0.raw text.raw\n";

    bool binary_write = true;
    BaseFloat learning_rate = -1;
    std::string learning_rates_str = "";
    std::string rename_output_nodes = "";

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("learning-rate", &learning_rate,
                "If supplied, all the learning rates of updatable components"
                "are set to this value.");
    po.Register("rename-output-nodes", &rename_output_nodes, "Rename one or"
                " more output nodes. For example, TODO");
    po.Register("scale-learning-rates", &learning_rates_str, "TODO");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = po.GetArg(1),
                raw_nnet_wxfilename = po.GetArg(2);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    if (rename_output_nodes != "") {
      RenameNodes(rename_output_nodes, &nnet);
    }

    if (learning_rate >= 0)
      SetLearningRate(learning_rate, &nnet);

    if (learning_rates_str != "") {
      std::vector<BaseFloat> learning_rates_std_vec;
      SplitStringToFloats(learning_rates_str, ",", false, &learning_rates_std_vec);
      Vector<BaseFloat> learning_rates(learning_rates_std_vec.size());
      for (int32 i = 0; i < learning_rates_std_vec.size(); i++)
        learning_rates(i) = learning_rates_std_vec[i];
      ScaleLearningRates(learning_rates, &nnet);
    }

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Copied raw neural net from " << raw_nnet_rxfilename
              << " to " << raw_nnet_wxfilename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
