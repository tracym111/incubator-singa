/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#ifndef SINGA_NEURALNET_NEURON_LAYER_POOLING_H_
#define SINGA_NEURALNET_NEURON_LAYER_POOLING_H_

#include <vector>
#include "singa/neuralnet/layer.h"
#include "singa/proto/job.pb.h"

namespace singa {

class PoolingLayer : public NeuronLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

 protected:
  int kernel_, pad_, stride_;
  int batchsize_, channels_, height_, width_, pooled_height_, pooled_width_;
  PoolingProto_PoolMethod pool_;
};
/**
 * Use book-keeping for BP following Caffe's pooling implementation
 */
class CPoolingLayer : public PoolingLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers);
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
 private:
  Blob<float> mask_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURON_LAYER_POOLING_H_
