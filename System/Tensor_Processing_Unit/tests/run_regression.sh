#!/usr/bin/env bash
#
# File: run_regression.sh
# Author: Functional TPU project
#
# One-command Questa regression driver for the Functional TPU testbenches.
#
# Each testbench is compiled into its OWN fresh work library and simulated in
# batch mode. Separate libraries are mandatory: the unit testbenches carry
# behavioral stubs (e.g. a fake Program_Memory / systolic-array accumulator)
# whose module names collide with the real RTL, so a shared library would pick
# up the wrong definition.
#
# ---------------------------------------------------------------------------
# USAGE
#   ./tests/run_regression.sh              # run all testbenches
#   ./tests/run_regression.sh 3 4 8        # run only Step3, Step4, Step8
#
# Run from anywhere; paths are resolved relative to this script's location.
#
# ---------------------------------------------------------------------------
# MACHINE GATE
#   This script only runs where Questa is installed (see QUESTA_DIR below). On
#   any other machine it prints a deferral notice and exits non-zero WITHOUT
#   running a simulator. This matches the CLAUDE.md rule: away from the Questa
#   PC, testbenches are still written/updated but verification is deferred here.
#
# ---------------------------------------------------------------------------
# HOW TO UPDATE THIS SCRIPT (adding a testbench or changing an RTL source list)
#   Every testbench is one `run_tb` block in the "TESTBENCH DEFINITIONS" section
#   below. To add a new testbench (e.g. Step 9):
#     1. Add a block:
#          run_tb 9 TB_Step9_Foo \
#            TPU/PATH/To/Dependency_A.v \
#            TPU/PATH/To/Dependency_B.v \
#            tests/TB_Step9_Foo.v
#        - First arg  = step number (used for subset selection on the CLI).
#        - Second arg = top-level module name (must equal the testbench module).
#        - Remaining  = RTL sources in dependency order, testbench LAST.
#     2. The compile list must match the "How to run" header inside the .v
#        testbench (each testbench documents its own required sources). Keep the
#        two in sync — the testbench header is the source of truth.
#   To change an existing testbench's dependencies, edit its source list in the
#   matching block. No other part of the script needs to change; subset
#   selection, logging, and the summary all key off the step number and top
#   name automatically.
#
# ---------------------------------------------------------------------------

set -u

# ---------- CONFIGURATION ---------- #
# Questa install (this is the only machine in the project with Questa). The
# presence of vsim.exe here is the machine gate.
QUESTA_DIR="C:/intelFPGA_lite/23.1std/questa_fse/win64"
# Documented Questa host (secondary identifier only; the gate is the path above).
QUESTA_HOST="CEC-EGB267-05"

# Project root = directory that contains this script's parent (tests/..).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"

# Simulation scratch lives OFF the OneDrive-synced Desktop to avoid the
# transient vopt file-lock seen there (see log.md 2026-06-30). Override with
# SIM_WORK=/some/path if desired.
SIM_WORK="${SIM_WORK:-${LOCALAPPDATA:-$PROJ}/tpu_sim}"
# ---------- END CONFIGURATION ---------- #

# ---------- MACHINE GATE ---------- #
if [ ! -x "$QUESTA_DIR/vsim.exe" ]; then
  echo "=============================================================="
  echo " Questa not found at: $QUESTA_DIR"
  echo " This machine cannot run the TPU regression."
  echo " Testbenches should still be written/updated for RTL changes,"
  echo " but simulation is DEFERRED to the Questa PC ($QUESTA_HOST)."
  echo " Record verification as PENDING in log.md."
  echo "=============================================================="
  exit 2
fi
export PATH="$QUESTA_DIR:$PATH"
# ---------- END MACHINE GATE ---------- #

# Requested step numbers (empty => all).
REQUESTED=("$@")

want_step () {
  # No args => run everything.
  [ "${#REQUESTED[@]}" -eq 0 ] && return 0
  local s="$1"
  local r
  for r in "${REQUESTED[@]}"; do
    [ "$r" = "$s" ] && return 0
  done
  return 1
}

# Accumulators for the final summary.
SUMMARY=()
OVERALL_OK=1

run_tb () {
  local step="$1"; shift
  local top="$1"; shift
  want_step "$step" || return 0

  local wl="$SIM_WORK/$top"
  rm -rf "$wl"; mkdir -p "$wl"

  echo "===== Step $step: $top ====="
  (
    cd "$wl" || exit 1
    vlib work >/dev/null 2>&1
    local src
    for src in "$@"; do
      if ! vlog -work work "$PROJ/$src" >> compile.log 2>&1; then
        echo "COMPILE-FAIL: $src"
        tail -25 compile.log
        exit 1
      fi
    done
    vsim -c -L altera_mf_ver -voptargs=+acc "work.$top" \
      -do "run -all; quit -f" > sim.log 2>&1
  )
  local rc=$?

  local result_line
  result_line="$(grep -iE "Results:" "$wl/sim.log" 2>/dev/null | tail -1)"

  if [ $rc -ne 0 ] || [ -z "$result_line" ]; then
    echo "  !! Step $step FAILED to complete (see $wl/compile.log, $wl/sim.log)"
    grep -iE "\*\* Error|\*\* Fatal|COMPILE-FAIL" "$wl/compile.log" "$wl/sim.log" 2>/dev/null | head -10
    SUMMARY+=("Step $step $top: ERROR")
    OVERALL_OK=0
    return
  fi

  echo "  ${result_line#\# }"
  SUMMARY+=("Step $step $top: ${result_line#\# }")
  # A per-test "Test N: FAIL" line is printed by every testbench regardless of
  # its summary wording ("X PASS, Y FAIL" vs. "X / Y tests passed"), so it is the
  # reliable cross-testbench failure signal.
  if grep -qiE "Test [0-9]+: FAIL" "$wl/sim.log"; then
    OVERALL_OK=0
    SUMMARY+=("  ^^ Step $step has FAILING test(s)")
  fi
}

# ============================================================= #
#                    TESTBENCH DEFINITIONS                      #
#  Source lists mirror the "How to run" header in each .v file. #
# ============================================================= #

run_tb 1 TB_Step1_Programmer \
  TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v \
  TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v \
  TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v \
  TPU/PROGRAMMER/Programmer.v \
  tests/TB_Step1_Programmer.v

run_tb spi TB_SPI_Slave \
  TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v \
  tests/TB_SPI_Slave.v

run_tb 1 TB_Step1_DataMemory \
  TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v \
  TPU/MEMORY/MEM_UNIT/Mem_Unit.v \
  TPU/MEMORY/Data_Memory.v \
  tests/TB_Step1_DataMemory.v

run_tb 2 TB_Step2_Feeder \
  TPU/PROGRAMMER/Feeder.v \
  tests/TB_Step2_Feeder.v

run_tb 3 TB_Step3_Controller \
  TPU/CONTROL/Controller.v \
  tests/TB_Step3_Controller.v

run_tb 4 TB_Step4_VectorProcessor \
  TPU/PROCESSING/Vector_Processor.v \
  tests/TB_Step4_VectorProcessor.v

run_tb 5 TB_Step5_Activator \
  TPU/PROCESSING/Activator.v \
  tests/TB_Step5_Activator.v

run_tb 6 TB_Step6_ALU \
  TPU/PROCESSING/ALU.v \
  tests/TB_Step6_ALU.v

run_tb pool TB_Pooler \
  TPU/PROCESSING/Pooler.v \
  tests/TB_Pooler.v

run_tb 7 TB_Step7_SystolicArray \
  TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v \
  TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v \
  TPU/SYSTOLIC_ARRAY/Systolic_Array.v \
  tests/TB_Step7_SystolicArray.v

run_tb 8 TB_Step8_FullSystem \
  TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v \
  TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v \
  TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v \
  TPU/PROGRAMMER/PROGRAM_MEMORY/Program_Memory.v \
  TPU/PROGRAMMER/Programmer.v \
  TPU/PROGRAMMER/Feeder.v \
  TPU/CONTROL/Controller.v \
  TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v \
  TPU/MEMORY/MEM_UNIT/Mem_Unit.v \
  TPU/MEMORY/Data_Memory.v \
  TPU/MEMORY/VECTOR_BUFFER/Vector_Buffer.v \
  TPU/PROCESSING/Activator.v \
  TPU/PROCESSING/ALU.v \
  TPU/PROCESSING/Vector_Processor.v \
  TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v \
  TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v \
  TPU/SYSTOLIC_ARRAY/Systolic_Array.v \
  TPU/TPU.v \
  tests/TB_Step8_FullSystem.v

# ============================================================= #

echo ""
echo "==================== REGRESSION SUMMARY ===================="
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo "==========================================================="
if [ "$OVERALL_OK" -eq 1 ]; then
  echo "ALL PASS"
  exit 0
else
  echo "FAILURES PRESENT"
  exit 1
fi
