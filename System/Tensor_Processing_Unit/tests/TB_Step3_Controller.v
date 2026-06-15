/* 
 * File: TB_Step3_Controller.v
 * Date: 2026-06-15
 *
 * Testbench for Build Step 3: Controller module.
 * Tests the Controller state machine using a behavioral Feeder stub and
 * stub idle signals for Vector_Processor and Systolic_Array.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. Open a terminal and cd to the project root:
 *        cd <path>/Tensor_Processing_Unit
 *
 *   2. Compile:
 *        vlib work
 *        vlog -work work TPU/CONTROL/Controller.v
 *        vlog -work work tests/TB_Step3_Controller.v
 *
 *   3. Simulate:
 *        vsim -voptargs=+acc -gui work.TB_Step3_Controller -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary prints total pass/fail counts.
 *   All 7 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: After reset — controller_idle HIGH, clear LOW
 *   Test 2: controller_start triggers clear for exactly one cycle
 *   Test 3: Invalid opcode → DECODE_ERROR (controller_idle never returns HIGH)
 *   Test 4: mult instruction — correct VP_A and VP_B execute-phase signals
 *   Test 5: relu instruction — correct VP_A execute-phase signals, VP_B not started
 *   Test 6: add instruction — correct VP_A and VP_B execute-phase signals
 *   Test 7: controller_idle returns HIGH after writeback completes
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step3_Controller;

// ---------------------------------------------------------------------------
// Clock / reset
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
// DUT signals
// ---------------------------------------------------------------------------
reg         controller_start;
reg [127:0] instruction;
reg         vector_idle_a;
reg         vector_idle_b;
reg         systolic_array_idle;

wire        controller_idle;
wire        clear;
wire        vector_start_a;
wire        vector_start_b;
wire [2:0]  vect_source_a;
wire [2:0]  vect_dest_a;
wire [15:0] mem_source_address_a;
wire [15:0] mem_dest_address_a;
wire [15:0] length_a;
wire [3:0]  dim0_a;
wire [3:0]  dim1_a;
wire        vector_start_b_out;
wire [2:0]  vect_source_b;
wire [2:0]  vect_dest_b;
wire [15:0] mem_source_address_b;
wire [15:0] length_b;
wire [3:0]  dim0_b;
wire [3:0]  dim1_b;
wire [2:0]  activator_funct;
wire [2:0]  alu_funct;
wire        systolic_array_start;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
Controller dut (
    .i_clk                  (clk),
    .i_trst                 (trst),
    .i_controller_start     (controller_start),
    .i_instruction          (instruction),
    .i_vector_idle_a        (vector_idle_a),
    .i_vector_idle_b        (vector_idle_b),
    .i_systolic_array_idle  (systolic_array_idle),
    .o_controller_idle      (controller_idle),
    .o_clear                (clear),
    .o_vector_start_a       (vector_start_a),
    .o_vect_source_a        (vect_source_a),
    .o_vect_dest_a          (vect_dest_a),
    .o_mem_source_address_a (mem_source_address_a),
    .o_mem_dest_address_a   (mem_dest_address_a),
    .o_length_a             (length_a),
    .o_dim0_a               (dim0_a),
    .o_dim1_a               (dim1_a),
    .o_vector_start_b       (vector_start_b_out),
    .o_vect_source_b        (vect_source_b),
    .o_vect_dest_b          (vect_dest_b),
    .o_mem_source_address_b (mem_source_address_b),
    .o_length_b             (length_b),
    .o_dim0_b               (dim0_b),
    .o_dim1_b               (dim1_b),
    .o_activator_funct      (activator_funct),
    .o_alu_funct            (alu_funct),
    .o_systolic_array_start (systolic_array_start)
);

// ---------------------------------------------------------------------------
// ISA constants
// ---------------------------------------------------------------------------
localparam [3:0] OP_MULT = 4'b1000;
localparam [3:0] OP_RELU = 4'b1001;
localparam [3:0] OP_ADD  = 4'b1010;
localparam [3:0] OP_BAD  = 4'b0111; // invalid opcode

// Controller source/dest encodings (must match Controller.v parameters)
localparam [2:0] VSRC_MEM     = 3'd0;
localparam [2:0] VSRC_SA_OUT  = 3'd1;
localparam [2:0] VSRC_VEC_BUF = 3'd2;
localparam [2:0] VDST_MEM     = 3'd0;
localparam [2:0] VDST_SA_A    = 3'd1;
localparam [2:0] VDST_SA_B    = 3'd2;
localparam [2:0] VDST_ACT     = 3'd3;
localparam [2:0] VDST_ALU_A   = 3'd4;
localparam [2:0] VDST_ALU_B   = 3'd5;
localparam [2:0] FUNCT_NOOP   = 3'd7;

// Build a mult instruction:
// opcode | rs1 | sz11 | sz12 | rs2 | sz21 | sz22 | rd | reserved | funct3
function [127:0] make_mult;
    input [15:0] rs1, rs2, rd;
    input [3:0]  sz11, sz12, sz21, sz22;
    make_mult = {OP_MULT, rs1, sz11, sz12, rs2, sz21, sz22, rd, 57'd0, 3'd0};
endfunction

// Build a relu instruction:
// opcode | rs1 | rd | len | reserved | funct3
function [127:0] make_relu;
    input [15:0] rs1, rd, len;
    make_relu = {OP_RELU, rs1, rd, len, 73'd0, 3'd0};
endfunction

// Build an add instruction:
// opcode | rs1 | rs2 | sz1 | sz2 | rd | reserved | funct3
function [127:0] make_add;
    input [15:0] rs1, rs2, rd;
    input [7:0]  sz1, sz2;
    make_add = {OP_ADD, rs1, rs2, sz1, sz2, rd, 57'd0, 3'd0};
endfunction

// ---------------------------------------------------------------------------
// Task: send one instruction to controller and wait for it to finish
// (simulates a complete Feeder-style handshake; VPs and SA are kept idle)
// ---------------------------------------------------------------------------
task send_instruction;
    input [127:0] instr;
    begin
        instruction = instr;
        @(posedge clk); #1;
        controller_start = 1'b1;
        @(posedge clk); #1;
        controller_start = 1'b0;
        // Wait for controller_idle to return HIGH (processing complete)
        while (!controller_idle)
            @(posedge clk); #1;
    end
endtask

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------
integer pass_count, fail_count;

task check;
    input        condition;
    input [63:0] test_num;
    begin
        if (condition)
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
// Stimulus
// ---------------------------------------------------------------------------
integer i;
integer clear_cycles;
integer vstart_a_seen, vstart_b_seen;
integer sa_start_seen;
reg [2:0] captured_vect_source_a, captured_vect_dest_a;
reg [15:0] captured_mem_src_a, captured_mem_dst_a;
reg [3:0] captured_dim0_a, captured_dim1_a;
reg [2:0] captured_vect_source_b, captured_vect_dest_b;
reg [15:0] captured_mem_src_b;
reg [3:0] captured_dim0_b, captured_dim1_b;
reg [2:0] captured_vect_source_b2, captured_vect_dest_b2;
reg [15:0] captured_length_a, captured_length_b;

initial
begin
    clk               = 0;
    trst              = 1;
    controller_start  = 0;
    instruction       = 128'd0;
    vector_idle_a     = 1'b1;
    vector_idle_b     = 1'b1;
    systolic_array_idle = 1'b1;
    pass_count        = 0;
    fail_count        = 0;

    // -----------------------------------------------------------------------
    // Test 1: After reset — controller_idle HIGH and clear LOW simultaneously
    // -----------------------------------------------------------------------
    do_reset;
    @(posedge clk); #1;
    check(controller_idle === 1'b1 && clear === 1'b0, 1);

    // -----------------------------------------------------------------------
    // Test 2: controller_start triggers clear for exactly one cycle
    // -----------------------------------------------------------------------
    do_reset;
    instruction = make_mult(16'h0010, 16'h0020, 16'h0030, 4'd2, 4'd3, 4'd3, 4'd4);

    // Pulse controller_start
    @(posedge clk); #1;
    controller_start = 1'b1;
    @(posedge clk); #1;
    controller_start = 1'b0;

    // Count cycles where clear is HIGH; should be exactly 1
    clear_cycles = 0;
    for (i = 0; i < 20; i = i + 1)
    begin
        if (clear === 1'b1)
            clear_cycles = clear_cycles + 1;
        @(posedge clk); #1;
    end
    check(clear_cycles === 1, 2);

    // Wait for controller to return to idle before next test
    while (!controller_idle) @(posedge clk); #1;

    // -----------------------------------------------------------------------
    // Test 3: Invalid opcode → DECODE_ERROR (controller_idle never returns)
    // -----------------------------------------------------------------------
    do_reset;
    instruction = {OP_BAD, 124'd0};
    @(posedge clk); #1;
    controller_start = 1'b1;
    @(posedge clk); #1;
    controller_start = 1'b0;

    // After reset, controller_idle was HIGH. After start it should go LOW and
    // stay LOW (stuck in DECODE_ERROR).
    begin : decode_error_block
        integer timeout;
        reg idle_returned;
        timeout      = 0;
        idle_returned = 0;
        while (timeout < 30)
        begin
            @(posedge clk); #1;
            if (controller_idle === 1'b1)
                idle_returned = 1;
            timeout = timeout + 1;
        end
        check(idle_returned === 0, 3);
    end

    // -----------------------------------------------------------------------
    // Test 4: mult instruction — correct VP_A and VP_B execute-phase signals
    // (sample signals in EXEC_VP_START / EXEC_VP_WAIT states)
    // -----------------------------------------------------------------------
    do_reset;
    // Prevent VPs and SA from immediately ack-ing so we can sample signals
    vector_idle_a     = 1'b0;
    vector_idle_b     = 1'b0;
    systolic_array_idle = 1'b0;

    instruction = make_mult(
        16'h0010, 16'h0020, 16'h0030,
        4'd2, 4'd3, 4'd3, 4'd4
    );

    @(posedge clk); #1;
    controller_start = 1'b1;
    @(posedge clk); #1;
    controller_start = 1'b0;

    // Advance to EXEC_VP_START then sample the combinational signals
    // CLEAR takes 1 cycle after start, EXEC_VP_START is next
    @(posedge clk); #1; // now in CLEAR
    @(posedge clk); #1; // now in EXEC_VP_START — sample combinational outputs

    captured_vect_source_a = vect_source_a;
    captured_vect_dest_a   = vect_dest_a;
    captured_mem_src_a     = mem_source_address_a;
    captured_dim0_a        = dim0_a;
    captured_dim1_a        = dim1_a;
    captured_vect_source_b = vect_source_b;
    captured_vect_dest_b   = vect_dest_b;
    captured_mem_src_b     = mem_source_address_b;
    captured_dim0_b        = dim0_b;
    captured_dim1_b        = dim1_b;

    check(
        (captured_vect_source_a === VSRC_MEM)  &&
        (captured_vect_dest_a   === VDST_SA_A) &&
        (captured_mem_src_a     === 16'h0010)  &&
        (captured_dim0_a        === 4'd2)      &&
        (captured_dim1_a        === 4'd3)      &&
        (captured_vect_source_b === VSRC_MEM)  &&
        (captured_vect_dest_b   === VDST_SA_B) &&
        (captured_mem_src_b     === 16'h0020)  &&
        (captured_dim0_b        === 4'd3)      &&
        (captured_dim1_b        === 4'd4),
        4
    );

    // Release stubs to let controller finish
    vector_idle_a     = 1'b1;
    vector_idle_b     = 1'b1;
    systolic_array_idle = 1'b1;
    while (!controller_idle) @(posedge clk); #1;

    // -----------------------------------------------------------------------
    // Test 5: relu — only VP_A started; VP_B vector_start never asserted
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a     = 1'b0;
    vector_idle_b     = 1'b1; // B is idle but should NOT be started
    systolic_array_idle = 1'b1;

    instruction = make_relu(16'h0050, 16'h0060, 16'd8);

    vstart_b_seen = 0;

    @(posedge clk); #1;
    controller_start = 1'b1;
    @(posedge clk); #1;
    controller_start = 1'b0;

    // Monitor vector_start_b for 10 cycles to confirm it is never asserted
    for (i = 0; i < 10; i = i + 1)
    begin
        if (vector_start_b_out === 1'b1)
            vstart_b_seen = 1;
        @(posedge clk); #1;
    end

    // Also sample VP_A execute-phase signals in EXEC_VP_START
    // (We were in CLEAR at cycle 2, EXEC_VP_START at cycle 3 — already past,
    //  but combinational signals are still valid in EXEC_VP_WAIT)
    captured_vect_source_a = vect_source_a;
    captured_vect_dest_a   = vect_dest_a;
    captured_mem_src_a     = mem_source_address_a;
    captured_length_a      = length_a;

    check(
        (vstart_b_seen === 0) &&
        (captured_vect_source_a === VSRC_MEM) &&
        (captured_vect_dest_a   === VDST_ACT) &&
        (captured_mem_src_a     === 16'h0050) &&
        (captured_length_a      === 16'd8),
        5
    );

    // Release VP_A to let controller finish
    vector_idle_a     = 1'b1;
    while (!controller_idle) @(posedge clk); #1;

    // -----------------------------------------------------------------------
    // Test 6: add — correct VP_A and VP_B execute-phase signals; check
    // elem_length = sz1 * sz2 = 3 * 4 = 12
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a     = 1'b0;
    vector_idle_b     = 1'b0;
    systolic_array_idle = 1'b1;

    instruction = make_add(16'h0070, 16'h0080, 16'h0090, 8'd3, 8'd4);

    @(posedge clk); #1;
    controller_start = 1'b1;
    @(posedge clk); #1;
    controller_start = 1'b0;

    @(posedge clk); #1; // CLEAR
    @(posedge clk); #1; // EXEC_VP_START — sample

    captured_vect_source_a  = vect_source_a;
    captured_vect_dest_a    = vect_dest_a;
    captured_mem_src_a      = mem_source_address_a;
    captured_length_a       = length_a;
    captured_vect_source_b2 = vect_source_b;
    captured_vect_dest_b2   = vect_dest_b;
    captured_mem_src_b      = mem_source_address_b;
    captured_length_b       = length_b;

    check(
        (captured_vect_source_a  === VSRC_MEM)   &&
        (captured_vect_dest_a    === VDST_ALU_A) &&
        (captured_mem_src_a      === 16'h0070)   &&
        (captured_length_a       === 16'd12)     &&
        (captured_vect_source_b2 === VSRC_MEM)   &&
        (captured_vect_dest_b2   === VDST_ALU_B) &&
        (captured_mem_src_b      === 16'h0080)   &&
        (captured_length_b       === 16'd12),
        6
    );

    // Release to finish
    vector_idle_a     = 1'b1;
    vector_idle_b     = 1'b1;
    while (!controller_idle) @(posedge clk); #1;

    // -----------------------------------------------------------------------
    // Test 7: controller_idle returns HIGH after writeback completes.
    // Hold VP_A non-idle from the start so the controller must wait through
    // EXEC_VP_WAIT and WB_VP_WAIT before returning to IDLE.
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a     = 1'b0;  // VP_A non-idle before instruction arrives
    vector_idle_b     = 1'b1;
    systolic_array_idle = 1'b1;

    instruction = make_relu(16'h00A0, 16'h00B0, 16'd5);

    @(posedge clk); #1;
    controller_start = 1'b1;
    @(posedge clk); #1;
    controller_start = 1'b0;

    // Advance past CLEAR and EXEC_VP_START into EXEC_VP_WAIT
    @(posedge clk); #1; // EXEC_VP_START
    @(posedge clk); #1; // EXEC_VP_WAIT — controller stuck because vector_idle_a=0

    // Release VP_A — controller can now advance through EXEC_VP_WAIT → WB
    vector_idle_a = 1'b1;

    // Wait up to 10 cycles to see controller_idle return HIGH
    begin : wb_test_block
        integer t;
        integer saw_idle;
        saw_idle = 0;
        t = 0;
        while (t < 10)
        begin
            @(posedge clk); #1;
            if (controller_idle === 1'b1)
                saw_idle = 1;
            t = t + 1;
        end
        check(saw_idle === 1, 7);
    end

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    #20;
    $display("-----------------------------");
    $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
    $display("-----------------------------");
    $finish;
end

endmodule
