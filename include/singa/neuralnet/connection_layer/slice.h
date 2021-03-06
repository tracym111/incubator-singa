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

#ifndef SINGA_NEURALNET_CONNECTION_LAYER_SLICE_H_
#define SINGA_NEURALNET_CONNECTION_LAYER_SLICE_H_

#include <vector>
#include "singa/neuralnet/layer.h"

namespace singa {
/**
 * Connect a single (src) layer with multiple (dst) layers.
 *
 * It slices the feature Blob (i.e., matrix) of the src layer on one dimension.
 * The sliced feature Blobs will be fed into dst layers.
 */
class SliceLayer : public ConnectionLayer {
 public:
  ~SliceLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const Blob<float>& data(const Layer* from) const override;
  const Blob<float>& grad(const Layer* from) const override;
  Blob<float>* mutable_data(const Layer* from) override;
  Blob<float>* mutable_grad(const Layer* from) override;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_CONNECTION_LAYER_SLICE_H_
