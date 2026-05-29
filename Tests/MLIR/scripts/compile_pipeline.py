#!/usr/bin/env python3
"""Run the educational MLIR pipeline and optionally build a demo executable.

This script shows the end-to-end example used in this workspace.

Key files and paths referenced by this driver (workspace-relative):
- Input sample: Tiny_NN_Recent.mlir (root of this workspace)
- Normalized intermediate: <work-dir>/normalized_input.mlir (generated)
- Lowering binary: build/bin/mlir-educational-example (default)
- Lowered LLVM-dialect output: <work-dir>/lowered_to_llvm.mlir (generated)
- LLVM IR / object: <work-dir>/lowered.ll, <work-dir>/lowered.o (generated)
- Host C++ wrapper (generated): <work-dir>/host_main.cpp
- Final executable (generated): <work-dir>/tiny_nn_demo

The primary lowering implementation lives in:
- lib/Conversion/MyHWToLLVM.cpp

Build and link configuration is in the top-level CMakeLists.txt. See
docs/COMPILATION_FLOW.md for a high-level explanation (docs/COMPILATION_FLOW.md).

The script keeps each stage visible:
1. Normalize the StableHLO sample into a tiny MLIR module.
2. Run the C++ lowering tool (the compiled `mlir-educational-example`).
3. Optionally translate the LLVM dialect output to LLVM IR and link a demo
    executable using system MLIR/LLVM tools (mlir-translate-19, llc-19,
    clang++-19).
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "build"
DEFAULT_BINARY = DEFAULT_BUILD_DIR / "bin" / "mlir-educational-example"
DEFAULT_MLIR_TRANSLATE = "/usr/bin/mlir-translate-19"
DEFAULT_LLC = "/usr/bin/llc-19"
DEFAULT_CLANGXX = "/usr/bin/clang++-19"

# Note: tools used below default to system-installed MLIR/LLVM 19 variants.
# If your distribution installs them in different locations, pass the
# `--mlir-translate`, `--llc`, and `--clangxx` flags to this script.

def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_tensor_type(type_text: str) -> tuple[list[int], str]:
    inner = type_text.strip()
    if not inner.startswith("tensor<") or not inner.endswith(">"):  # pragma: no cover
        raise ValueError(f"unsupported tensor type: {type_text}")
    payload = inner[len("tensor<") : -1]
    pieces = payload.split("x")
    if len(pieces) == 1:
        return [], pieces[0]
    return [int(part) for part in pieces[:-1]], pieces[-1]


def memref_type(shape: list[int], element_type: str) -> str:
    if not shape:
        return f"memref<{element_type}>"
    return f"memref<{ 'x'.join(str(dim) for dim in shape) }x{element_type}>"


def normalize_stablehlo(input_path: Path, output_path: Path) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    func_line = next(line for line in lines if "func.func public @main" in line)
    signature_match = re.search(r"\((.*)\) -> \((.*)\)", func_line)
    if signature_match is None:
        raise ValueError("could not parse the input function signature")

    input_text, output_text = signature_match.groups()
    input_items = [item.strip() for item in input_text.split(",") if item.strip()]

    arg_specs: list[tuple[str, list[int], str]] = []
    for item in input_items:
        item_match = re.search(r"%(\w+):\s*(tensor<[^>]+>)", item)
        if item_match is None:
            raise ValueError(f"could not parse argument item: {item}")
        name, type_text = item_match.groups()
        shape, element_type = parse_tensor_type(type_text)
        arg_specs.append((name, shape, element_type))

    output_match = re.search(r"tensor<([^>]+)>", output_text)
    if output_match is None:
        raise ValueError("could not parse the output tensor type")
    output_payload = output_match.group(1)
    output_parts = output_payload.split("x")
    output_shape = [int(part) for part in output_parts[:-1]]
    output_element = output_parts[-1]

    # The transformation is intentionally narrow and mirrors the sample file.
    # Adding new supported StableHLO ops should happen here first.
    simplified: list[str] = []
    simplified.append("module {")
    simplified.append(
        "  func.func @kernel(" + ", ".join(
            f"%{name}: {memref_type(shape, element_type)}"
            for name, shape, element_type in arg_specs
        ) + f", %out: {memref_type(output_shape, output_element)}) {{"
    )
    simplified.append("    %cst = arith.constant 1.000000e+00 : f32")
    simplified.append(
        "    %0 = \"myhw.transpose\"(%arg0) {permutation = [1, 0]} : (memref<4x1xf32>) -> memref<1x4xf32>"
    )
    simplified.append(
        "    %1 = \"myhw.broadcast\"(%cst) {broadcast_dims = []} : (f32) -> memref<4xf32>"
    )
    simplified.append(
        "    %2 = \"myhw.elementwise\"(%arg1, %1) {kind = \"multiply\"} : (memref<4xf32>, memref<4xf32>) -> memref<4xf32>"
    )
    simplified.append(
        "    %3 = \"myhw.matmul\"(%arg4, %0) : (memref<1x1xf32>, memref<1x4xf32>) -> memref<1x4xf32>"
    )
    simplified.append(
        "    %4 = \"myhw.broadcast\"(%cst) {broadcast_dims = []} : (f32) -> memref<1x4xf32>"
    )
    simplified.append(
        "    %5 = \"myhw.elementwise\"(%4, %3) {kind = \"multiply\"} : (memref<1x4xf32>, memref<1x4xf32>) -> memref<1x4xf32>"
    )
    simplified.append(
        "    %6 = \"myhw.broadcast\"(%2) {broadcast_dims = [0]} : (memref<4xf32>) -> memref<1x4xf32>"
    )
    simplified.append(
        "    %7 = \"myhw.elementwise\"(%6, %5) {kind = \"add\"} : (memref<1x4xf32>, memref<1x4xf32>) -> memref<1x4xf32>"
    )
    simplified.append(
        "    %8 = \"myhw.transpose\"(%arg2) {permutation = [1, 0]} : (memref<1x4xf32>) -> memref<4x1xf32>"
    )
    simplified.append(
        "    %9 = \"myhw.broadcast\"(%cst) {broadcast_dims = []} : (f32) -> memref<1xf32>"
    )
    simplified.append(
        "    %10 = \"myhw.elementwise\"(%arg3, %9) {kind = \"multiply\"} : (memref<1xf32>, memref<1xf32>) -> memref<1xf32>"
    )
    simplified.append(
        "    %11 = \"myhw.matmul\"(%7, %8) : (memref<1x4xf32>, memref<4x1xf32>) -> memref<1x1xf32>"
    )
    simplified.append(
        "    %12 = \"myhw.broadcast\"(%cst) {broadcast_dims = []} : (f32) -> memref<1x1xf32>"
    )
    simplified.append(
        "    %13 = \"myhw.elementwise\"(%12, %11) {kind = \"multiply\"} : (memref<1x1xf32>, memref<1x1xf32>) -> memref<1x1xf32>"
    )
    simplified.append(
        "    %14 = \"myhw.broadcast\"(%10) {broadcast_dims = [0]} : (memref<1xf32>) -> memref<1x1xf32>"
    )
    simplified.append(
        "    %15 = \"myhw.elementwise\"(%14, %13) {kind = \"add\"} : (memref<1x1xf32>, memref<1x1xf32>) -> memref<1x1xf32>"
    )
    simplified.append("    memref.copy %15, %out : memref<1x1xf32> to memref<1x1xf32>")
    simplified.append("    return")
    simplified.append("  }")
    simplified.append("}")

    output_path.write_text("\n".join(simplified) + "\n", encoding="utf-8")


## Writes the host wrapper C++ file (for example: <work-dir>/host_main.cpp).
## The generated file will be compiled and linked with the lowered object
## produced from the LLVM IR translation.
def write_host_program(path: Path, kernel_name: str) -> None:
    path.write_text(
        f"""#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "mlir/ExecutionEngine/CRunnerUtils.h"

extern "C" void {kernel_name}(float *arg0_base, float *arg0_data, int64_t arg0_offset,
                              int64_t arg0_size0, int64_t arg0_size1,
                              int64_t arg0_stride0, int64_t arg0_stride1,
                              float *arg1_base, float *arg1_data, int64_t arg1_offset,
                              int64_t arg1_size0, int64_t arg1_stride0,
                              float *arg2_base, float *arg2_data, int64_t arg2_offset,
                              int64_t arg2_size0, int64_t arg2_size1,
                              int64_t arg2_stride0, int64_t arg2_stride1,
                              float *arg3_base, float *arg3_data, int64_t arg3_offset,
                              int64_t arg3_size0, int64_t arg3_stride0,
                              float *arg4_base, float *arg4_data, int64_t arg4_offset,
                              int64_t arg4_size0, int64_t arg4_size1,
                              int64_t arg4_stride0, int64_t arg4_stride1,
                              float *out_base, float *out_data, int64_t out_offset,
                              int64_t out_size0, int64_t out_size1,
                              int64_t out_stride0, int64_t out_stride1);

int main() {{
  auto *arg0Buffer = static_cast<float *>(std::malloc(sizeof(float) * 4));
  auto *arg1Buffer = static_cast<float *>(std::malloc(sizeof(float) * 4));
  auto *arg2Buffer = static_cast<float *>(std::malloc(sizeof(float) * 4));
  auto *arg3Buffer = static_cast<float *>(std::malloc(sizeof(float) * 1));
  auto *arg4Buffer = static_cast<float *>(std::malloc(sizeof(float) * 1));
  auto *outBuffer = static_cast<float *>(std::malloc(sizeof(float) * 1));

  for (int i = 0; i < 4; ++i) {{
    arg0Buffer[i] = static_cast<float>(i + 1);
    arg1Buffer[i] = 1.0f;
  }}
  arg2Buffer[0] = 1.0f;
  arg2Buffer[1] = 2.0f;
  arg2Buffer[2] = 3.0f;
  arg2Buffer[3] = 4.0f;
  arg3Buffer[0] = 1.0f;
  arg4Buffer[0] = 1.0f;
  outBuffer[0] = 0.0f;

    StridedMemRefType<float, 2> arg0{{arg0Buffer, arg0Buffer, 0, {{4, 1}}, {{1, 1}}}};
    StridedMemRefType<float, 1> arg1{{arg1Buffer, arg1Buffer, 0, {{4}}, {{1}}}};
    StridedMemRefType<float, 2> arg2{{arg2Buffer, arg2Buffer, 0, {{1, 4}}, {{4, 1}}}};
    StridedMemRefType<float, 1> arg3{{arg3Buffer, arg3Buffer, 0, {{1}}, {{1}}}};
    StridedMemRefType<float, 2> arg4{{arg4Buffer, arg4Buffer, 0, {{1, 1}}, {{1, 1}}}};
    StridedMemRefType<float, 2> out{{outBuffer, outBuffer, 0, {{1, 1}}, {{1, 1}}}};

    {kernel_name}(arg0.basePtr, arg0.data, arg0.offset, arg0.sizes[0], arg0.sizes[1],
                                 arg0.strides[0], arg0.strides[1],
                                 arg1.basePtr, arg1.data, arg1.offset, arg1.sizes[0], arg1.strides[0],
                                 arg2.basePtr, arg2.data, arg2.offset, arg2.sizes[0], arg2.sizes[1],
                                 arg2.strides[0], arg2.strides[1],
                                 arg3.basePtr, arg3.data, arg3.offset, arg3.sizes[0], arg3.strides[0],
                                 arg4.basePtr, arg4.data, arg4.offset, arg4.sizes[0], arg4.sizes[1],
                                 arg4.strides[0], arg4.strides[1],
                                 out.basePtr, out.data, out.offset, out.sizes[0], out.sizes[1],
                                 out.strides[0], out.strides[1]);
    std::printf("result = %f\\n", outBuffer[0]);

  std::free(arg0Buffer);
  std::free(arg1Buffer);
  std::free(arg2Buffer);
  std::free(arg3Buffer);
  std::free(arg4Buffer);
  std::free(outBuffer);
  return 0;
}}
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the educational MLIR pipeline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_BUILD_DIR / "pipeline")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--mlir-translate", default=DEFAULT_MLIR_TRANSLATE)
    parser.add_argument("--llc", default=DEFAULT_LLC)
    parser.add_argument("--clangxx", default=DEFAULT_CLANGXX)
    parser.add_argument("--emit-executable", action="store_true")
    args = parser.parse_args()

    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized_mlir = work_dir / "normalized_input.mlir"
    lowered_mlir = work_dir / "lowered_to_llvm.mlir"
    llvm_ir = work_dir / "lowered.ll"
    object_file = work_dir / "lowered.o"
    host_cpp = work_dir / "host_main.cpp"
    executable = work_dir / "tiny_nn_demo"

    normalize_stablehlo(args.input, normalized_mlir)
    run([str(args.binary), "--input", str(normalized_mlir), "--output", str(lowered_mlir)])

    if not args.emit_executable:
        print(f"Wrote LLVM dialect MLIR to {lowered_mlir}")
        return 0

    run([args.mlir_translate, "--mlir-to-llvmir", str(lowered_mlir), "-o", str(llvm_ir)])
    run([args.llc, str(llvm_ir), "-filetype=obj", "-o", str(object_file)])
    write_host_program(host_cpp, "kernel")
    run([
        args.clangxx,
        str(host_cpp),
        str(object_file),
        "-I/usr/lib/llvm-19/include",
        "-L/usr/lib/llvm-19/lib",
        "-o",
        str(executable),
    ])
    print(f"Wrote LLVM dialect MLIR to {lowered_mlir}")
    print(f"Wrote executable to {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
