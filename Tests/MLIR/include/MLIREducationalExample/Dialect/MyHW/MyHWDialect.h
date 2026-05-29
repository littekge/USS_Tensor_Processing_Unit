#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::educational::myhw {

class MyHWDialect : public Dialect {
public:
  explicit MyHWDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "myhw"; }
  void initialize();
};

class TransposeOp : public Op<TransposeOp, OpTrait::OneOperand,
                              OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "myhw.transpose"; }
  static ArrayRef<StringRef> getAttributeNames() {
    static constexpr StringRef names[] = {"permutation"};
    return names;
  }
  static void build(OpBuilder &builder, OperationState &state, Value input,
                    Type resultType, ArrayRef<int64_t> permutation);

  Value input() { return getOperand(); }
  ArrayAttr permutationAttr() { return (*this)->getAttrOfType<ArrayAttr>("permutation"); }
};

class BroadcastOp : public Op<BroadcastOp, OpTrait::OneOperand,
                              OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "myhw.broadcast"; }
  static ArrayRef<StringRef> getAttributeNames() {
    static constexpr StringRef names[] = {"broadcast_dims"};
    return names;
  }
  static void build(OpBuilder &builder, OperationState &state, Value input,
                    Type resultType, ArrayRef<int64_t> broadcastDims);

  Value input() { return getOperand(); }
  ArrayAttr broadcastDimsAttr() {
    return (*this)->getAttrOfType<ArrayAttr>("broadcast_dims");
  }
};

class ElementwiseOp : public Op<ElementwiseOp, OpTrait::NOperands<2>::Impl,
                                OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "myhw.elementwise"; }
  static ArrayRef<StringRef> getAttributeNames() {
    static constexpr StringRef names[] = {"kind"};
    return names;
  }
  static void build(OpBuilder &builder, OperationState &state, Value lhs,
                    Value rhs, Type resultType, StringRef kind);

  Value lhs() { return getOperand(0); }
  Value rhs() { return getOperand(1); }
  StringAttr kindAttr() { return (*this)->getAttrOfType<StringAttr>("kind"); }
};

class MatMulOp : public Op<MatMulOp, OpTrait::NOperands<2>::Impl,
                           OpTrait::OneResult> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "myhw.matmul"; }
  static ArrayRef<StringRef> getAttributeNames() {
    return ArrayRef<StringRef>();
  }
  static void build(OpBuilder &builder, OperationState &state, Value lhs,
                    Value rhs, Type resultType);

  Value lhs() { return getOperand(0); }
  Value rhs() { return getOperand(1); }
};

std::unique_ptr<Pass> createStableHloLiteToMyHWPass();
std::unique_ptr<Pass> createMyHWToLLVMReadyPass();

} // namespace mlir::educational::myhw
