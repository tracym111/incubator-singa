#include "neuralnet/layer.h"
#include "myproto.pb.h"

namespace singa {
class HiddenLayer : public NeuronLayer {
 public:
  ~HiddenLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

// please fill HiddenLayer class declaration
}