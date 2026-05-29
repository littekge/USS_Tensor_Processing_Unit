#include "MLIREducationalExample/Dialect/MyHW/MyHWDialect.h"

namespace mlir::educational::myhw {

MyHWDialect::MyHWDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MyHWDialect>()) {
  initialize();
}

void MyHWDialect::initialize() {
  addOperations<TransposeOp, BroadcastOp, ElementwiseOp, MatMulOp>();
}

void TransposeOp::build(OpBuilder &builder, OperationState &state, Value input,
                        Type resultType, ArrayRef<int64_t> permutation) {
  state.addOperands(input);
  state.addTypes(resultType);
  SmallVector<Attribute> attrs;
  attrs.reserve(permutation.size());
  for (int64_t value : permutation)
    attrs.push_back(builder.getI64IntegerAttr(value));
  state.addAttribute("permutation", builder.getArrayAttr(attrs));
}

void BroadcastOp::build(OpBuilder &builder, OperationState &state, Value input,
                        Type resultType, ArrayRef<int64_t> broadcastDims) {
  state.addOperands(input);
  state.addTypes(resultType);
  SmallVector<Attribute> attrs;
  attrs.reserve(broadcastDims.size());
  for (int64_t value : broadcastDims)
    attrs.push_back(builder.getI64IntegerAttr(value));
  state.addAttribute("broadcast_dims", builder.getArrayAttr(attrs));
}

void ElementwiseOp::build(OpBuilder &builder, OperationState &state, Value lhs,
                          Value rhs, Type resultType, StringRef kind) {
  state.addOperands({lhs, rhs});
  state.addTypes(resultType);
  state.addAttribute("kind", builder.getStringAttr(kind));
}

void MatMulOp::build(OpBuilder &builder, OperationState &state, Value lhs,
                     Value rhs, Type resultType) {
  state.addOperands({lhs, rhs});
  state.addTypes(resultType);
}

} // namespace mlir::educational::myhw
