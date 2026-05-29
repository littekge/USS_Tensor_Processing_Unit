#include "MLIREducationalExample/Dialect/MyHW/MyHWDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::educational::myhw;

namespace {

static SmallVector<int64_t> readIntArray(Attribute attr) {
  SmallVector<int64_t> values;
  auto arrayAttr = dyn_cast_or_null<ArrayAttr>(attr);
  if (!arrayAttr)
    return values;
  values.reserve(arrayAttr.size());
  for (Attribute element : arrayAttr) {
    auto integerAttr = dyn_cast<IntegerAttr>(element);
    if (!integerAttr)
      continue;
    values.push_back(integerAttr.getInt());
  }
  return values;
}

static Value constantIndex(PatternRewriter &rewriter, Location loc, int64_t v) {
  return rewriter.create<arith::ConstantIndexOp>(loc, v);
}

static void emitAllIndices(PatternRewriter &rewriter, Location loc,
                           ArrayRef<int64_t> shape,
                           SmallVectorImpl<Value> &indices, unsigned dim,
                           function_ref<void(OpBuilder &, Location, ValueRange)>
                               bodyBuilder) {
  if (dim == shape.size()) {
    bodyBuilder(rewriter, loc, indices);
    return;
  }

  for (int64_t index = 0; index < shape[dim]; ++index) {
    indices.push_back(constantIndex(rewriter, loc, index));
    emitAllIndices(rewriter, loc, shape, indices, dim + 1, bodyBuilder);
    indices.pop_back();
  }
}

static MemRefType getMemRefResultType(Type type) {
  return cast<MemRefType>(type);
}

struct TransposeLowering : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = getMemRefResultType(op->getResult(0).getType());
    auto sourceType = dyn_cast<MemRefType>(op.input().getType());
    if (!resultType || !sourceType)
      return failure();

    SmallVector<int64_t> permutation = readIntArray(op.permutationAttr());
    if (permutation.size() != resultType.getRank())
      return failure();

    Value result = rewriter.create<memref::AllocOp>(op.getLoc(), resultType);
    ArrayRef<int64_t> shape = resultType.getShape();
    SmallVector<Value> indices;
    emitAllIndices(rewriter, op.getLoc(), shape, indices, 0,
                   [&](OpBuilder &bodyBuilder, Location loc, ValueRange ivs) {
                     SmallVector<Value> sourceIndices(sourceType.getRank());
                     for (size_t i = 0; i < permutation.size(); ++i)
                       sourceIndices[permutation[i]] = ivs[i];
                     Value loaded = bodyBuilder.create<memref::LoadOp>(
                         loc, op.input(), sourceIndices);
                     bodyBuilder.create<memref::StoreOp>(loc, loaded, result,
                                                           ivs);
                   });

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BroadcastLowering : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = getMemRefResultType(op->getResult(0).getType());
    if (!resultType)
      return failure();

    SmallVector<int64_t> broadcastDims = readIntArray(op.broadcastDimsAttr());
    auto sourceMemRef = dyn_cast<MemRefType>(op.input().getType());
    Value source = op.input();
    bool sourceIsMemRef = static_cast<bool>(sourceMemRef);

    if (sourceIsMemRef && broadcastDims.size() != sourceMemRef.getRank())
      return failure();

    Value result = rewriter.create<memref::AllocOp>(op.getLoc(), resultType);
    ArrayRef<int64_t> shape = resultType.getShape();
    SmallVector<Value> indices;
    emitAllIndices(rewriter, op.getLoc(), shape, indices, 0,
                   [&](OpBuilder &bodyBuilder, Location loc, ValueRange ivs) {
                     Value element;
                     if (sourceIsMemRef) {
                       SmallVector<Value> sourceIndices(sourceMemRef.getRank());
                       for (size_t i = 0; i < broadcastDims.size(); ++i)
                         sourceIndices[i] = ivs[broadcastDims[i]];
                       element = bodyBuilder.create<memref::LoadOp>(loc, source,
                                                              sourceIndices);
                     } else {
                       element = source;
                     }
                     bodyBuilder.create<memref::StoreOp>(loc, element, result,
                                                           ivs);
                   });

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ElementwiseLowering : public OpRewritePattern<ElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = getMemRefResultType(op->getResult(0).getType());
    auto lhsType = dyn_cast<MemRefType>(op.lhs().getType());
    auto rhsType = dyn_cast<MemRefType>(op.rhs().getType());
    if (!resultType || !lhsType || !rhsType)
      return failure();

    Value result = rewriter.create<memref::AllocOp>(op.getLoc(), resultType);
    ArrayRef<int64_t> shape = resultType.getShape();
    StringRef kind = op.kindAttr() ? op.kindAttr().getValue() : "add";

    SmallVector<Value> indices;
    emitAllIndices(rewriter, op.getLoc(), shape, indices, 0,
                   [&](OpBuilder &bodyBuilder, Location loc, ValueRange ivs) {
                     SmallVector<Value> localIndices(ivs.begin(), ivs.end());
                     Value lhs = bodyBuilder.create<memref::LoadOp>(loc, op.lhs(),
                                                                    localIndices);
                     Value rhs = bodyBuilder.create<memref::LoadOp>(loc, op.rhs(),
                                                                    localIndices);
                     Value computed;
                     if (kind == "multiply")
                       computed = bodyBuilder.create<arith::MulFOp>(loc, lhs, rhs).getResult();
                     else
                       computed = bodyBuilder.create<arith::AddFOp>(loc, lhs, rhs).getResult();
                     bodyBuilder.create<memref::StoreOp>(loc, computed, result,
                                                           ivs);
                   });

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MatMulLowering : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = getMemRefResultType(op->getResult(0).getType());
    auto lhsType = dyn_cast<MemRefType>(op.lhs().getType());
    auto rhsType = dyn_cast<MemRefType>(op.rhs().getType());
    if (!resultType || !lhsType || !rhsType)
      return failure();
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        resultType.getRank() != 2)
      return failure();

    Value result = rewriter.create<memref::AllocOp>(op.getLoc(), resultType);
    auto resultShape = resultType.getShape();

    SmallVector<Value> outerIndices;
    emitAllIndices(rewriter, op.getLoc(), SmallVector<int64_t>{resultShape[0], resultShape[1]},
                   outerIndices, 0,
                   [&](OpBuilder &bodyBuilder, Location loc, ValueRange ivs) {
                     Value zero = bodyBuilder.create<arith::ConstantFloatOp>(
                         loc, APFloat(0.0f), bodyBuilder.getF32Type()).getResult();
                     Value sum = zero;

                     for (int64_t kIndex = 0; kIndex < lhsType.getShape()[1]; ++kIndex) {
                       Value kValue = bodyBuilder.create<arith::ConstantIndexOp>(loc, kIndex);
                       SmallVector<Value> lhsIndices{ivs[0], kValue};
                       SmallVector<Value> rhsIndices{kValue, ivs[1]};
                       Value lhs = bodyBuilder.create<memref::LoadOp>(loc, op.lhs(),
                                                                       lhsIndices);
                       Value rhs = bodyBuilder.create<memref::LoadOp>(loc, op.rhs(),
                                                                       rhsIndices);
                       Value product = bodyBuilder.create<arith::MulFOp>(loc, lhs, rhs).getResult();
                       sum = bodyBuilder.create<arith::AddFOp>(loc, sum, product).getResult();
                     }

                     bodyBuilder.create<memref::StoreOp>(loc, sum, result, ivs);
                   });

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MyHWToLLVMReadyPass
    : public PassWrapper<MyHWToLLVMReadyPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TransposeLowering, BroadcastLowering, ElementwiseLowering,
                 MatMulLowering>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::educational::myhw {

std::unique_ptr<Pass> createMyHWToLLVMReadyPass() {
  return std::make_unique<MyHWToLLVMReadyPass>();
}

} // namespace mlir::educational::myhw
