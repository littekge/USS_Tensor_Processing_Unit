#include "MLIREducationalExample/Dialect/MyHW/MyHWDialect.h"

#include "mlir/Pass/Pass.h"

namespace mlir::educational::myhw {
namespace {

struct StableHloLiteToMyHWPass
    : public PassWrapper<StableHloLiteToMyHWPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // The Python driver performs the StableHLO-lite normalization step.
    // This pass stays in the pipeline as a readable place to add checks or
    // future front-end transformations.
  }
};

} // namespace

std::unique_ptr<Pass> createStableHloLiteToMyHWPass() {
  return std::make_unique<StableHloLiteToMyHWPass>();
}

} // namespace mlir::educational::myhw
