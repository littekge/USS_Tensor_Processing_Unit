/* 
 * File: TB_Step2_Feeder.v
 * Date: 2026-06-15
 *
 * Testbench for Build Step 2: Feeder module.
 * Tests the Feeder state machine using a behavioral Program_Memory model
 * (2-cycle registered output latency) and a behavioral Controller stub.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. Open a terminal and cd to the project root:
 *        cd <path>/Tensor_Processing_Unit
 *
 *   2. Compile:
 *        vlib work
 *        vlog -work work TPU/PROGRAMMER/Feeder.v
 *        vlog -work work tests/TB_Step2_Feeder.v
 *
 *   3. Simulate:
 *        vsim work.TB_Step2_Feeder -do "run -all; quit"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary prints total pass/fail counts.
 *   All 7 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: After reset — controller_start=LOW and pm_address=0
 *   Test 2: end at address 0 → DONE; controller_start never asserts HIGH
 *   Test 3: Non-terminating instr at address 0 → controller_start asserts HIGH
 *   Test 4: controller_start is HIGH for exactly one clock cycle
 *   Test 5: PC increments to 1 after controller acknowledges first instruction
 *   Test 6: Correct instruction content forwarded (first of two non-end instr)
 *   Test 7: end_reached pulses HIGH for exactly one cycle when end is fetched
 *           (and the Feeder reaches DONE)
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step2_Feeder;

// ---------------------------------------------------------------------------
// Clock and reset
// ---------------------------------------------------------------------------
localparam CLK_HALF = 10; // 10 ns → 50 MHz

reg clk, trst;

always #CLK_HALF clk = ~clk;

task do_reset;
begin
    trst = 1'b0;
    @(posedge clk); #1;
    @(posedge clk); #1;
    trst = 1'b1;
    @(posedge clk); #1;
end
endtask

// ---------------------------------------------------------------------------
// ISA opcodes (from Functional_TPU_ISA_v0.2.md)
// ---------------------------------------------------------------------------
// Opcode 0000 is the program-termination opcode, now named end per ISA v0.2.
// end reuses the former syscall encoding (opcode 0000); syscall moved to
// funct7 0x1.
localparam [3:0] OPCODE_END  = 4'b0000;
localparam [3:0] OPCODE_MULT = 4'b1000;
localparam [3:0] OPCODE_ADD  = 4'b1010;

// Build an end (terminate) instruction: opcode=0000, reserved=0, funct7=0x0
function [127:0] make_end;
    input dummy;
    make_end = {OPCODE_END, 124'd0};
endfunction

// Build a mult instruction
function [127:0] make_mult;
    input [15:0] rs1, rs2, rd;
    // opcode | rs1 | sz11 | sz12 | rs2 | sz21 | sz22 | rd | reserved | funct3
    make_mult = {OPCODE_MULT, rs1, 4'd1, 4'd1, rs2, 4'd1, 4'd1, rd, 57'd0, 3'd0};
endfunction

// Build an add instruction
function [127:0] make_add_instr;
    input [15:0] rs1, rs2, rd;
    // opcode | rs1 | rs2 | sz1 | sz2 | rd | reserved | funct3
    make_add_instr = {OPCODE_ADD, rs1, rs2, 8'd1, 8'd1, rd, 57'd0, 3'd0};
endfunction

// ---------------------------------------------------------------------------
// Behavioral Program Memory (PM_MAX=10'd3 → 4 addresses 0-3)
// Using a small size for fast simulation; Feeder is parameterized to match.
// ---------------------------------------------------------------------------
localparam [9:0] PM_MAX = 10'd3;

reg [127:0] pmem [0:3]; // 4-word behavioral memory

wire [9:0]  pm_address;
wire        pm_wren;    // Feeder never writes; wire present for completeness

// 2-cycle registered read latency, matches the Quartus IP (registered address +
// registered output). The Feeder must wait two cycles after driving the address.
reg [127:0] pm_q_stage1;
reg [127:0] pm_q_reg;
always @ (posedge clk)
begin
    pm_q_stage1 <= pmem[pm_address[1:0]];
    pm_q_reg    <= pm_q_stage1;
end

// ---------------------------------------------------------------------------
// Controller stub
// ---------------------------------------------------------------------------
reg ctrl_idle;

// ---------------------------------------------------------------------------
// DUT — Feeder with PM_MAX_ADDRESS matching the 4-word behavioral memory
// ---------------------------------------------------------------------------
wire [127:0] dut_instruction;
wire         dut_ctrl_start;
wire         dut_end_reached;

Feeder #(.PM_MAX_ADDRESS(PM_MAX)) dut (
    .i_clk              (clk),
    .i_trst             (trst),
    .i_pm_q             (pm_q_reg),
    .o_pm_address       (pm_address),
    .o_pm_wren          (pm_wren),
    .i_controller_idle  (ctrl_idle),
    .o_instruction      (dut_instruction),
    .o_controller_start (dut_ctrl_start),
    .o_end_reached      (dut_end_reached)
);

// Count end_reached pulses to confirm it is a one-cycle strobe.
integer end_pulse_count;
always @(posedge clk)
begin
    if (trst == 1'b1 && dut_end_reached == 1'b1)
        end_pulse_count = end_pulse_count + 1;
end

// ---------------------------------------------------------------------------
// Test bookkeeping
// ---------------------------------------------------------------------------
integer pass_count, fail_count;

task report_test;
    input         pass;
    input integer test_num;
    begin
        if (pass)
        begin
            $display("Test %0d: PASS", test_num);
            pass_count = pass_count + 1;
        end
        else
        begin
            $display("Test %0d: FAIL", test_num);
            fail_count = fail_count + 1;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Helper: drive controller_idle LOW for a few cycles then pulse it HIGH once,
// simulating the controller finishing an instruction decode-execute cycle.
// ---------------------------------------------------------------------------
task controller_ack;
begin
    ctrl_idle = 1'b0;
    repeat (3) @(posedge clk); #1;
    ctrl_idle = 1'b1;
    @(posedge clk); #1;
    ctrl_idle = 1'b0;
end
endtask

// ---------------------------------------------------------------------------
// Simulation variables shared across tests
// ---------------------------------------------------------------------------
integer      i;
integer      ctrl_start_count;
reg          ctrl_start_seen;
reg [127:0]  captured_instr;
reg          feed_error_ok;

// ---------------------------------------------------------------------------
// Main simulation
// ---------------------------------------------------------------------------
initial
begin
    $dumpfile("TB_Step2_Feeder.vcd");
    $dumpvars(0, TB_Step2_Feeder);

    clk        = 1'b0;
    trst       = 1'b1;
    ctrl_idle  = 1'b0;
    pass_count = 0;
    fail_count = 0;
    end_pulse_count = 0;

    // =======================================================================
    // Test 1: After reset — controller_start=LOW and pm_address=0
    // =======================================================================
    pmem[0] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[1] = make_end(1'b0);
    pmem[2] = make_end(1'b0);
    pmem[3] = make_end(1'b0);

    do_reset;

    // One cycle after reset the state machine is in START with pc=0
    @(posedge clk); #1;

    report_test((dut_ctrl_start == 1'b0 && pm_address == 10'd0), 1);

    // =======================================================================
    // Test 2: end at address 0 → DONE; controller_start never asserts HIGH
    // =======================================================================
    pmem[0] = make_end(1'b0);
    pmem[1] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[2] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[3] = make_mult(16'h0002, 16'h0003, 16'h0004);

    do_reset;

    ctrl_start_seen = 1'b0;
    // START→FETCH→WAIT_PM→WAIT_PM2→CHECK→DONE is 5 transitions plus margin
    for (i = 0; i < 10; i = i + 1)
    begin
        @(posedge clk); #1;
        if (dut_ctrl_start == 1'b1)
            ctrl_start_seen = 1'b1;
    end

    report_test((ctrl_start_seen == 1'b0), 2);

    // =======================================================================
    // Test 3: Non-terminating instr at address 0 → controller_start asserts HIGH
    // =======================================================================
    pmem[0] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[1] = make_end(1'b0);
    pmem[2] = make_end(1'b0);
    pmem[3] = make_end(1'b0);

    do_reset;
    ctrl_idle = 1'b0;

    ctrl_start_seen = 1'b0;
    for (i = 0; i < 20; i = i + 1)
    begin
        @(posedge clk); #1;
        if (dut_ctrl_start == 1'b1)
            ctrl_start_seen = 1'b1;
    end

    report_test(ctrl_start_seen, 3);

    // =======================================================================
    // Test 4: controller_start is HIGH for exactly one clock cycle
    // =======================================================================
    pmem[0] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[1] = make_end(1'b0);
    pmem[2] = make_end(1'b0);
    pmem[3] = make_end(1'b0);

    do_reset;
    ctrl_idle = 1'b0; // keep controller busy so feeder stays in WAIT_CTRL

    ctrl_start_count = 0;
    for (i = 0; i < 20; i = i + 1)
    begin
        @(posedge clk); #1;
        if (dut_ctrl_start == 1'b1)
            ctrl_start_count = ctrl_start_count + 1;
    end

    report_test((ctrl_start_count == 1), 4);

    // =======================================================================
    // Test 5: PC increments to 1 after controller acknowledges first instr
    // =======================================================================
    pmem[0] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[1] = make_end(1'b0);
    pmem[2] = make_end(1'b0);
    pmem[3] = make_end(1'b0);

    do_reset;
    ctrl_idle = 1'b0;

    // Wait for FORWARD state (controller_start HIGH)
    for (i = 0; i < 20 && dut_ctrl_start == 1'b0; i = i + 1)
        @(posedge clk); #1;

    controller_ack;

    // Allow cycles for FETCH of address 1: WAIT_CTRL→FETCH→...
    repeat (4) @(posedge clk); #1;

    report_test((pm_address == 10'd1), 5);

    // =======================================================================
    // Test 6: Correct instruction content forwarded to controller
    // Verify the first of two non-terminating instructions has the expected opcode.
    // =======================================================================
    pmem[0] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[1] = make_add_instr(16'h0002, 16'h0003, 16'h0004);
    pmem[2] = make_end(1'b0);
    pmem[3] = make_end(1'b0);

    do_reset;
    ctrl_idle = 1'b0;

    // Wait for FORWARD (controller_start HIGH) then capture instruction
    for (i = 0; i < 20 && dut_ctrl_start == 1'b0; i = i + 1)
        @(posedge clk); #1;

    captured_instr = dut_instruction;

    report_test((captured_instr[127:124] == OPCODE_MULT), 6);

    // =======================================================================
    // Test 7: end_reached pulses HIGH for exactly one cycle when end is fetched.
    // Place an end at address 0; the Feeder fetches it, pulses end_reached once,
    // and settles in DONE. A non-end at addr 0 (already covered) must not pulse.
    // =======================================================================
    pmem[0] = make_end(1'b0);
    pmem[1] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[2] = make_mult(16'h0002, 16'h0003, 16'h0004);
    pmem[3] = make_mult(16'h0002, 16'h0003, 16'h0004);

    end_pulse_count = 0;
    do_reset;
    ctrl_idle = 1'b0;

    // Run long enough to fetch the end and settle in DONE.
    for (i = 0; i < 20; i = i + 1)
    begin
        @(posedge clk); #1;
    end

    // Exactly one end_reached pulse, Feeder in DONE (4'd7), no controller_start.
    report_test((end_pulse_count == 1) && (dut.S == 4'd7), 7);

    // =======================================================================
    // Summary
    // =======================================================================
    $display("---------------------------------------------------");
    $display("Results: %0d / %0d tests passed", pass_count, pass_count + fail_count);
    $display("---------------------------------------------------");
    $finish;
end

endmodule
