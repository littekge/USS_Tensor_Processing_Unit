#include "MLIREducationalExample/Dialect/MyHW/MyHWDialect.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::educational::myhw;

namespace {

struct PipelineOptions {
  std::string inputFile;
  std::string outputFile;
  bool dumpIntermediate = false;
};

void registerDialects(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                      scf::SCFDialect, LLVM::LLVMDialect, MyHWDialect>();
}

static OwningOpRef<ModuleOp> parseInputMLIR(MLIRContext &context,
                                            StringRef fileName) {
  llvm::SourceMgr sourceMgr;
  auto bufferOrError = llvm::MemoryBuffer::getFile(fileName);
  if (!bufferOrError) {
    llvm::errs() << "failed to open input file: " << fileName << '\n';
    return nullptr;
  }
  sourceMgr.AddNewSourceBuffer(std::move(*bufferOrError), SMLoc());
  context.allowUnregisteredDialects();
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

void buildPassManager(PassManager &pm, bool dumpIntermediate) {
  pm.enableVerifier(true);
  pm.addPass(createStableHloLiteToMyHWPass());
  pm.addPass(createMyHWToLLVMReadyPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (dumpIntermediate)
    pm.enableIRPrinting();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);

  llvm::cl::opt<std::string> inputFile("input", llvm::cl::desc("StableHLO-lite input file"),
                                       llvm::cl::Required);
  llvm::cl::opt<std::string> outputFile("output", llvm::cl::desc("Output LLVM dialect MLIR file"),
                                        llvm::cl::init(""));
  llvm::cl::opt<bool> dumpIntermediate("dump-intermediate", llvm::cl::desc("Print each pass result"),
                                       llvm::cl::init(false));

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Educational Example\n");

  MLIRContext context;
  registerDialects(context);

  auto module = parseInputMLIR(context, inputFile);
  if (!module) {
    return 1;
  }

  PassManager pm(&context);
  buildPassManager(pm, dumpIntermediate);
  if (failed(pm.run(*module))) {
    llvm::errs() << "pass pipeline failed\n";
    return 1;
  }

  if (outputFile.empty()) {
    module->print(llvm::outs());
    llvm::outs() << '\n';
    return 0;
  }

  std::error_code errorCode;
  llvm::raw_fd_ostream out(outputFile, errorCode);
  if (errorCode) {
    llvm::errs() << errorCode.message() << '\n';
    return 1;
  }
  module->print(out);
  out.flush();
  return 0;
}
