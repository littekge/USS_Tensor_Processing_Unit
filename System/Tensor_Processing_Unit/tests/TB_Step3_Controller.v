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
 *   All 18 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: After reset — controller_idle HIGH, clear LOW
 *   Test 2: controller_start triggers clear for exactly one cycle
 *   Test 3: Invalid opcode → DECODE_ERROR (controller_idle never returns HIGH)
 *   Test 4: mult instruction — correct VP_A and VP_B execute-phase signals
 *   Test 5: relu instruction — correct VP_A execute-phase signals, VP_B not started
 *   Test 6: add instruction — correct VP_A and VP_B execute-phase signals
 *   Test 7: controller_idle returns HIGH after writeback completes
 *   Test 8: systolic_array_start is asserted for mult but not for relu/add
 *   Test 9: function-code decode — relu→activator RELU, add→ALU ADD,
 *           mult→both NOOP
 *   Test 10: mult requant params (M0/n) decoded and asserted to VP_A during
 *            writeback (VSRC_SA_OUT), and held at 0 during the execute decode
 *   Test 11 (v0.5): accumulator_clear pulses exactly once after a mult
 *            writeback (funct3 0x0) and is suppressed for multip (funct3 0x1);
 *            both variants decode and run to completion.
 *   Test 12 (v0.5): stride (opcode 1111) latches im1/im2 into ld1/ld2 and has
 *            no execute/writeback phase (no vector_start pulses).
 *   Test 13 (v0.5): after a stride, a MUL presents ld1 on leading_dimension_a
 *            and ld2 on leading_dimension_b during the src1/src2 load, and ld2
 *            on leading_dimension_a during writeback.
 *   Test 14 (v0.6): window (WINCONFIG) latches all eleven descriptor registers
 *            and runs no execute/writeback phase.
 *   Test 15 (v0.6): im2col decode — VSRC_MEM_WIN -> VDST_MEM at dest, pooler
 *            NO-OP, skips writeback (one vector_start_a pulse only).
 *   Test 16 (v0.6): max execute decode — VSRC_MEM_WIN -> VDST_POOL from src1,
 *            pooler function MAX.
 *   Test 17 (v0.6): max writeback decode — VSRC_VEC_BUF -> VDST_MEM at dest,
 *            length = chans*outh*outw (24 for the preceding window).
 *   Test 18 (v0.7): move decode — SHAPE opcode funct3 0x1 routes VSRC_MEM ->
 *            VDST_MEM (contiguous copy) with source = rs1, dest = SHAPE dest,
 *            length = aux; pooler NO-OP; no windowed source; skips writeback
 *            (exactly one vector_start_a pulse). The funct3 0x0 im2col path is
 *            unaffected (Test 15).
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
wire [23:0] mem_source_address_a;
wire [23:0] mem_dest_address_a;
wire [15:0] length_a;
wire [3:0]  dim0_a;
wire [3:0]  dim1_a;
wire [7:0]  scale_a;
wire [7:0]  shift_a;
wire        vector_start_b_out;
wire [2:0]  vect_source_b;
wire [2:0]  vect_dest_b;
wire [23:0] mem_source_address_b;
wire [15:0] length_b;
wire [3:0]  dim0_b;
wire [3:0]  dim1_b;
wire [2:0]  activator_funct;
wire [2:0]  alu_funct;
wire        systolic_array_start;
wire        accumulator_clear;
wire [23:0] leading_dimension_a;
wire [23:0] leading_dimension_b;
wire [2:0]  pooler_funct;
wire [11:0] win_chans_o, win_inh_o, win_inw_o, win_outh_o, win_outw_o;
wire [3:0]  win_winh_o, win_winw_o, win_strh_o, win_strw_o, win_padh_o, win_padw_o;

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
    .o_scale_a              (scale_a),
    .o_shift_a              (shift_a),
    .o_vector_start_b       (vector_start_b_out),
    .o_vect_source_b        (vect_source_b),
    .o_vect_dest_b          (vect_dest_b),
    .o_mem_source_address_b (mem_source_address_b),
    .o_length_b             (length_b),
    .o_dim0_b               (dim0_b),
    .o_dim1_b               (dim1_b),
    .o_leading_dimension_a  (leading_dimension_a),
    .o_leading_dimension_b  (leading_dimension_b),
    .o_activator_funct      (activator_funct),
    .o_alu_funct            (alu_funct),
    .o_systolic_array_start (systolic_array_start),
    .o_accumulator_clear    (accumulator_clear),
    .o_pooler_funct         (pooler_funct),
    .o_win_chans            (win_chans_o),
    .o_win_inh              (win_inh_o),
    .o_win_inw              (win_inw_o),
    .o_win_outh             (win_outh_o),
    .o_win_outw             (win_outw_o),
    .o_win_winh             (win_winh_o),
    .o_win_winw             (win_winw_o),
    .o_win_strh             (win_strh_o),
    .o_win_strw             (win_strw_o),
    .o_win_padh             (win_padh_o),
    .o_win_padw             (win_padw_o)
);

// ---------------------------------------------------------------------------
// ISA constants
// ---------------------------------------------------------------------------
localparam [3:0] OP_MULT   = 4'b1000;
localparam [3:0] OP_RELU   = 4'b1001;
localparam [3:0] OP_ADD    = 4'b1010;
localparam [3:0] OP_MAX    = 4'b1100;
localparam [3:0] OP_IM2COL = 4'b1101;
localparam [3:0] OP_WINDOW = 4'b1110;
localparam [3:0] OP_STRIDE = 4'b1111;
localparam [3:0] OP_BAD    = 4'b0111; // invalid opcode

// MUL funct3 sub-codes (per ISA Instruction Set Listing)
localparam [2:0] F3_MULT   = 3'h0;
localparam [2:0] F3_MULTIP = 3'h1;

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
localparam [2:0] VDST_POOL    = 3'd6;
localparam [2:0] VSRC_MEM_WIN = 3'd3;
localparam [2:0] FUNCT_NOOP   = 3'd7;
localparam [2:0] FUNCT_RELU   = 3'd0;
localparam [2:0] FUNCT_ADD    = 3'd0;
localparam [2:0] FUNCT_MAX    = 3'd0;

// Count systolic_array_start pulses (used by Test 8)
integer sa_start_count;
always @(posedge clk)
    if (trst === 1'b1 && systolic_array_start === 1'b1)
        sa_start_count = sa_start_count + 1;

// Count accumulator_clear pulses (used by Test 11)
integer acc_clear_count;
always @(posedge clk)
    if (trst === 1'b1 && accumulator_clear === 1'b1)
        acc_clear_count = acc_clear_count + 1;

// Count vector_start_a pulses (used by the v0.6 windowed tests to prove
// im2col runs one VP phase — execute only — while relu/add/max run two).
integer vsa_count;
always @(posedge clk)
    if (trst === 1'b1 && vector_start_a === 1'b1)
        vsa_count = vsa_count + 1;

// Build a MUL instruction (ISA v0.5 MUL format): opcode[127:124] |
// rs1[123:100] | sz11[99:96] | sz12[95:92] | rs2[91:68] | sz21[67:64] |
// sz22[63:60] | rd[59:36] | M0[35:28] | n[27:20] | reserved[19:3] |
// funct3[2:0]. funct3 selects mult (0x0) vs multip (0x1).
function [127:0] make_mul_f3;
    input [23:0] rs1, rs2, rd;
    input [3:0]  sz11, sz12, sz21, sz22;
    input [7:0]  scale, shift;
    input [2:0]  f3;
    make_mul_f3 = {OP_MULT, rs1, sz11, sz12, rs2, sz21, sz22, rd,
                   scale, shift, 17'd0, f3};
endfunction

// mult with explicit requant params (funct3 = mult).
function [127:0] make_mult_rq;
    input [23:0] rs1, rs2, rd;
    input [3:0]  sz11, sz12, sz21, sz22;
    input [7:0]  scale, shift;
    make_mult_rq = make_mul_f3(rs1, rs2, rd, sz11, sz12, sz21, sz22,
                               scale, shift, F3_MULT);
endfunction

// mult with zero requant params (existing tests do not inspect scale/shift).
function [127:0] make_mult;
    input [23:0] rs1, rs2, rd;
    input [3:0]  sz11, sz12, sz21, sz22;
    make_mult = make_mul_f3(rs1, rs2, rd, sz11, sz12, sz21, sz22,
                            8'd0, 8'd0, F3_MULT);
endfunction

// multip (funct3 = multip) — identical layout, does not clear the accumulator.
function [127:0] make_multip;
    input [23:0] rs1, rs2, rd;
    input [3:0]  sz11, sz12, sz21, sz22;
    make_multip = make_mul_f3(rs1, rs2, rd, sz11, sz12, sz21, sz22,
                              8'd0, 8'd0, F3_MULTIP);
endfunction

// Build a stride instruction (ISA v0.5 CONFIG format): opcode[127:124] |
// im1[123:100] | im2[99:76] | reserved[75:3] | funct3[2:0] = 0.
function [127:0] make_stride;
    input [23:0] im1, im2;
    make_stride = {OP_STRIDE, im1, im2, 76'd0};
endfunction

// Build a relu instruction (ISA v0.4 ACT format):
// opcode[127:124] | rs1[123:100] | rd[99:76] | len[75:60] | reserved[59:0].
function [127:0] make_relu;
    input [23:0] rs1, rd;
    input [15:0] len;
    make_relu = {OP_RELU, rs1, rd, len, 60'd0};
endfunction

// Build an add instruction (ISA v0.4 ELEM format):
// opcode[127:124] | rs1[123:100] | rs2[99:76] | sz1[75:68] | sz2[67:60] |
// rd[59:36] | reserved[35:0] (funct3 in bits 2:0 = 0).
function [127:0] make_add;
    input [23:0] rs1, rs2, rd;
    input [7:0]  sz1, sz2;
    make_add = {OP_ADD, rs1, rs2, sz1, sz2, rd, 36'd0};
endfunction

// Build a window instruction (ISA v0.6 W-format): opcode[127:124] |
// inh[123:112] | inw[111:100] | chans[99:88] | outh[87:76] | outw[75:64] |
// winh[63:60] | winw[59:56] | strh[55:52] | strw[51:48] | padh[47:44] |
// padw[43:40] | reserved[39:3] | funct3[2:0] = 0.
function [127:0] make_window;
    input [11:0] inh, inw, chans, outh, outw;
    input [3:0]  winh, winw, strh, strw, padh, padw;
    make_window = {OP_WINDOW, inh, inw, chans, outh, outw,
                   winh, winw, strh, strw, padh, padw, 37'd0, 3'd0};
endfunction

// Build an im2col instruction (ISA v0.6 A-format):
// opcode[127:124] | src1[123:100] | dest[99:76] | aux[75:60] | reserved | funct3.
function [127:0] make_im2col;
    input [23:0] src1, dest;
    make_im2col = {OP_IM2COL, src1, dest, 16'd0, 57'd0, 3'd0};
endfunction

// Build a max instruction (ISA v0.6 A-format, same layout as im2col).
function [127:0] make_max;
    input [23:0] src1, dest;
    make_max = {OP_MAX, src1, dest, 16'd0, 57'd0, 3'd0};
endfunction

// Build a move instruction (ISA v0.7 SHAPE A-format): shares OP_IM2COL with
// im2col, distinguished by funct3 0x1. aux (bits 75:60) carries len.
// opcode[127:124] | src1[123:100] | dest[99:76] | aux[75:60] | reserved | funct3.
localparam [2:0] F3_MOVE = 3'h1;
function [127:0] make_move;
    input [23:0] src1, dest;
    input [15:0] len;
    make_move = {OP_IM2COL, src1, dest, len, 57'd0, F3_MOVE};
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
reg [23:0] captured_mem_src_a, captured_mem_dst_a;
reg [3:0] captured_dim0_a, captured_dim1_a;
reg [2:0] captured_vect_source_b, captured_vect_dest_b;
reg [23:0] captured_mem_src_b;
reg [3:0] captured_dim0_b, captured_dim1_b;
reg [2:0] captured_vect_source_b2, captured_vect_dest_b2;
reg [15:0] captured_length_a, captured_length_b;
reg mult_saw_sa, relu_saw_sa, add_saw_sa;
reg [2:0] cap_relu_act, cap_add_alu, cap_mult_act, cap_mult_alu;
reg [7:0] cap_scale_wb, cap_shift_wb;
reg       wb_captured, exec_rq_zero;
integer   mult_acc_clears, multip_acc_clears;
integer   stride_vstart_seen;
reg       ld_exec_a_ok, ld_exec_b_ok, ld_wb_a_ok;
integer   window_vstart_seen;
reg       im2col_route_ok, im2col_saw_wb_src;
reg       max_exec_ok, max_wb_ok;
reg [2:0] cap_im2col_pool, cap_max_pool_exec;
reg [15:0] cap_max_wb_len;
reg       move_route_ok, move_saw_wb_src, move_saw_win;
reg [2:0] cap_move_pool;
reg [15:0] cap_move_len;

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
    instruction = make_mult(24'h000010, 24'h000020, 24'h000030, 4'd2, 4'd3, 4'd3, 4'd4);

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
        24'h000010, 24'h000020, 24'h000030,
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
        (captured_mem_src_a     === 24'h000010)  &&
        (captured_dim0_a        === 4'd2)      &&
        (captured_dim1_a        === 4'd3)      &&
        (captured_vect_source_b === VSRC_MEM)  &&
        (captured_vect_dest_b   === VDST_SA_B) &&
        (captured_mem_src_b     === 24'h000020)  &&
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

    instruction = make_relu(24'h000050, 24'h000060, 16'd8);

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
        (captured_mem_src_a     === 24'h000050) &&
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

    instruction = make_add(24'h000070, 24'h000080, 24'h000090, 8'd3, 8'd4);

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
        (captured_mem_src_a      === 24'h000070)   &&
        (captured_length_a       === 16'd12)     &&
        (captured_vect_source_b2 === VSRC_MEM)   &&
        (captured_vect_dest_b2   === VDST_ALU_B) &&
        (captured_mem_src_b      === 24'h000080)   &&
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

    instruction = make_relu(24'h0000A0, 24'h0000B0, 16'd5);

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
    // Test 8: systolic_array_start asserted for mult, not for relu/add.
    // Keep all idle stubs HIGH so each instruction runs to completion.
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;

    sa_start_count = 0;
    send_instruction(make_mult(24'h000010, 24'h000020, 24'h000030, 4'd2, 4'd2, 4'd2, 4'd2));
    mult_saw_sa = (sa_start_count > 0);

    sa_start_count = 0;
    send_instruction(make_relu(24'h000050, 24'h000060, 16'd4));
    relu_saw_sa = (sa_start_count > 0);

    sa_start_count = 0;
    send_instruction(make_add(24'h000070, 24'h000080, 24'h000090, 8'd2, 8'd2));
    add_saw_sa = (sa_start_count > 0);

    check((mult_saw_sa === 1'b1) && (relu_saw_sa === 1'b0) &&
          (add_saw_sa === 1'b0), 8);

    // -----------------------------------------------------------------------
    // Test 9: function-code decode. Hold a VP non-idle to sample the
    // combinational function outputs during the execute phase.
    // -----------------------------------------------------------------------
    // relu -> activator RELU
    do_reset;
    vector_idle_a = 1'b0; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    instruction = make_relu(24'h000050, 24'h000060, 16'd4);
    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    @(posedge clk); #1; // CLEAR
    @(posedge clk); #1; // EXEC_VP_START
    cap_relu_act = activator_funct;
    vector_idle_a = 1'b1;
    while (!controller_idle) @(posedge clk); #1;

    // add -> ALU ADD
    do_reset;
    vector_idle_a = 1'b0; vector_idle_b = 1'b0; systolic_array_idle = 1'b1;
    instruction = make_add(24'h000070, 24'h000080, 24'h000090, 8'd2, 8'd2);
    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    @(posedge clk); #1; // CLEAR
    @(posedge clk); #1; // EXEC_VP_START
    cap_add_alu = alu_funct;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1;
    while (!controller_idle) @(posedge clk); #1;

    // mult -> both NOOP
    do_reset;
    vector_idle_a = 1'b0; vector_idle_b = 1'b0; systolic_array_idle = 1'b0;
    instruction = make_mult(24'h000010, 24'h000020, 24'h000030, 4'd2, 4'd2, 4'd2, 4'd2);
    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    @(posedge clk); #1; // CLEAR
    @(posedge clk); #1; // EXEC_VP_START
    cap_mult_act = activator_funct;
    cap_mult_alu = alu_funct;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    while (!controller_idle) @(posedge clk); #1;

    check((cap_relu_act === FUNCT_RELU) && (cap_add_alu === FUNCT_ADD) &&
          (cap_mult_act === FUNCT_NOOP) && (cap_mult_alu === FUNCT_NOOP), 9);

    // -----------------------------------------------------------------------
    // Test 10: mult requant params (M0/n) are decoded and asserted to VP_A
    // during the writeback step, and are NOT asserted during the execute step.
    //   The writeback decode is uniquely identified by vect_source_a ==
    //   VSRC_SA_OUT (VP_A reading the systolic array outputs); the execute
    //   decode routes VP_A to VDST_SA_A with requant params left at 0.
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    instruction = make_mult_rq(24'h000010, 24'h000020, 24'h000030,
                               4'd2, 4'd2, 4'd2, 4'd2, 8'd7, 8'd5);
    wb_captured  = 1'b0;
    exec_rq_zero = 1'b1;
    cap_scale_wb = 8'd0;
    cap_shift_wb = 8'd0;

    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;

    // Sample requant outputs across the whole instruction: capture during
    // writeback, confirm zero during the SA-load execute decode.
    while (!controller_idle)
    begin
        @(posedge clk); #1;
        if (vect_source_a === VSRC_MEM && vect_dest_a === VDST_SA_A)
            if (scale_a !== 8'd0 || shift_a !== 8'd0) exec_rq_zero = 1'b0;
        if (vect_source_a === VSRC_SA_OUT)
        begin
            cap_scale_wb = scale_a;
            cap_shift_wb = shift_a;
            wb_captured  = 1'b1;
        end
    end

    check((wb_captured  === 1'b1) &&
          (cap_scale_wb === 8'd7) &&
          (cap_shift_wb === 8'd5) &&
          (exec_rq_zero === 1'b1), 10);

    // -----------------------------------------------------------------------
    // Test 11 (v0.5): accumulator_clear pulses once after a mult writeback and
    // is suppressed for multip. Both variants decode and run to completion.
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;

    acc_clear_count = 0;
    send_instruction(make_mult(24'h000010, 24'h000020, 24'h000030,
                               4'd2, 4'd2, 4'd2, 4'd2));
    mult_acc_clears = acc_clear_count;

    acc_clear_count = 0;
    send_instruction(make_multip(24'h000010, 24'h000020, 24'h000030,
                                 4'd2, 4'd2, 4'd2, 4'd2));
    multip_acc_clears = acc_clear_count;

    check((mult_acc_clears === 1) && (multip_acc_clears === 0), 11);

    // -----------------------------------------------------------------------
    // Test 12 (v0.5): stride latches im1/im2 into ld1/ld2 and does not run an
    // execute or writeback phase (no vector_start pulses).
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;

    instruction        = make_stride(24'h000005, 24'h00000A);
    stride_vstart_seen = 0;
    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    while (!controller_idle)
    begin
        if (vector_start_a === 1'b1 || vector_start_b_out === 1'b1)
            stride_vstart_seen = 1;
        @(posedge clk); #1;
    end
    check((dut.ld1 === 24'h000005) && (dut.ld2 === 24'h00000A) &&
          (stride_vstart_seen === 0), 12);

    // -----------------------------------------------------------------------
    // Test 13 (v0.5): after the Test 12 stride (ld1=5, ld2=10), a MUL presents
    // ld1 on leading_dimension_a and ld2 on leading_dimension_b during the
    // src1/src2 load, and ld2 on leading_dimension_a during writeback. The ld
    // registers persist across instructions (no intervening stride/reset).
    // -----------------------------------------------------------------------
    ld_exec_a_ok = 1'b0;
    ld_exec_b_ok = 1'b0;
    ld_wb_a_ok   = 1'b0;
    instruction  = make_mult(24'h000010, 24'h000020, 24'h000030,
                             4'd2, 4'd2, 4'd2, 4'd2);
    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    while (!controller_idle)
    begin
        @(posedge clk); #1;
        if (vect_source_a === VSRC_MEM && vect_dest_a === VDST_SA_A)
        begin
            if (leading_dimension_a === 24'h000005) ld_exec_a_ok = 1'b1;
            if (leading_dimension_b === 24'h00000A) ld_exec_b_ok = 1'b1;
        end
        if (vect_source_a === VSRC_SA_OUT)
        begin
            if (leading_dimension_a === 24'h00000A) ld_wb_a_ok = 1'b1;
        end
    end
    check((ld_exec_a_ok === 1'b1) && (ld_exec_b_ok === 1'b1) &&
          (ld_wb_a_ok === 1'b1), 13);

    // -----------------------------------------------------------------------
    // Test 14 (v0.6): window (WINCONFIG) latches all eleven descriptor
    // registers and has no execute/writeback phase (no vector_start pulses).
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;

    instruction        = make_window(12'd28, 12'd28, 12'd6, 12'd24, 12'd24,
                                     4'd5, 4'd5, 4'd1, 4'd1, 4'd0, 4'd0);
    window_vstart_seen = 0;
    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    while (!controller_idle)
    begin
        if (vector_start_a === 1'b1 || vector_start_b_out === 1'b1)
            window_vstart_seen = 1;
        @(posedge clk); #1;
    end
    check((dut.inh   === 12'd28) && (dut.inw  === 12'd28) &&
          (dut.chans === 12'd6)  && (dut.outh === 12'd24) &&
          (dut.outw  === 12'd24) && (dut.winh === 4'd5)   &&
          (dut.winw  === 4'd5)   && (dut.strh === 4'd1)   &&
          (dut.strw  === 4'd1)   && (dut.padh === 4'd0)   &&
          (dut.padw  === 4'd0)   && (window_vstart_seen === 0),
          14);

    // -----------------------------------------------------------------------
    // Test 15 (v0.6): im2col decode — VP_A gathers (VSRC_MEM_WIN) into device
    // memory (VDST_MEM) at dest, pooler funct is NO-OP, and it skips writeback
    // (exactly one vector_start_a pulse, VP_A never reads SA outputs or the
    // vector buffer).
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b0; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    instruction        = make_im2col(24'h000100, 24'h000200);
    im2col_route_ok    = 1'b0;
    im2col_saw_wb_src  = 1'b0;
    cap_im2col_pool    = FUNCT_MAX; // seed with wrong value; must become NOOP
    vsa_count          = 0;

    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    @(posedge clk); #1; // CLEAR
    @(posedge clk); #1; // EXEC_VP_START — sample execute decode
    if ((vect_source_a === VSRC_MEM_WIN) && (vect_dest_a === VDST_MEM) &&
        (mem_source_address_a === 24'h000100) &&
        (mem_dest_address_a === 24'h000200))
        im2col_route_ok = 1'b1;
    cap_im2col_pool = pooler_funct;
    vector_idle_a = 1'b1; // release
    while (!controller_idle)
    begin
        @(posedge clk); #1;
        if (vect_source_a === VSRC_SA_OUT || vect_source_a === VSRC_VEC_BUF)
            im2col_saw_wb_src = 1'b1;
    end
    check((im2col_route_ok === 1'b1) && (cap_im2col_pool === FUNCT_NOOP) &&
          (im2col_saw_wb_src === 1'b0) && (vsa_count === 1), 15);

    // -----------------------------------------------------------------------
    // Test 16 (v0.6): max execute decode — VP_A streams (VSRC_MEM_WIN) to the
    // Pooler (VDST_POOL) from src1, and the pooler function is MAX.
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b0; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    instruction       = make_max(24'h000300, 24'h000400);
    max_exec_ok       = 1'b0;
    cap_max_pool_exec = FUNCT_NOOP; // seed wrong; must become MAX

    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    @(posedge clk); #1; // CLEAR
    @(posedge clk); #1; // EXEC_VP_START — sample execute decode
    if ((vect_source_a === VSRC_MEM_WIN) && (vect_dest_a === VDST_POOL) &&
        (mem_source_address_a === 24'h000300))
        max_exec_ok = 1'b1;
    cap_max_pool_exec = pooler_funct;
    vector_idle_a = 1'b1;
    while (!controller_idle) @(posedge clk); #1;
    check((max_exec_ok === 1'b1) && (cap_max_pool_exec === FUNCT_MAX), 16);

    // -----------------------------------------------------------------------
    // Test 17 (v0.6): max writeback decode — after streaming to the Pooler,
    // VP_A copies the pooled results from the vector buffer (VSRC_VEC_BUF) to
    // dest (VDST_MEM) with length = chans*outh*outw. A preceding window sets
    // chans=2, outh=3, outw=4 -> length 24.
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b1; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    send_instruction(make_window(12'd6, 12'd6, 12'd2, 12'd3, 12'd4,
                                 4'd2, 4'd2, 4'd2, 4'd2, 4'd0, 4'd0));

    vector_idle_a = 1'b0; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    instruction    = make_max(24'h000300, 24'h000400);
    max_wb_ok      = 1'b0;
    cap_max_wb_len = 16'd0;

    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    // Sweep the whole instruction; capture the writeback decode.
    while (!controller_idle)
    begin
        @(posedge clk); #1;
        if ((vect_source_a === VSRC_VEC_BUF) && (vect_dest_a === VDST_MEM) &&
            (mem_dest_address_a === 24'h000400))
        begin
            max_wb_ok      = 1'b1;
            cap_max_wb_len = length_a;
        end
        vector_idle_a = 1'b1; // let it proceed through both phases
    end
    check((max_wb_ok === 1'b1) && (cap_max_wb_len === 16'd24), 17);

    // -----------------------------------------------------------------------
    // Test 18 (v0.7): move decode — SHAPE opcode (OP_IM2COL) with funct3 0x1
    // routes a contiguous copy VSRC_MEM -> VDST_MEM, source = rs1, dest = SHAPE
    // dest, length = aux; pooler NO-OP; never engages the windowed source; and
    // skips writeback (exactly one vector_start_a pulse, no SA/vector-buffer
    // read decode). Distinct from im2col (funct3 0x0, Test 15) on the same
    // opcode.
    // -----------------------------------------------------------------------
    do_reset;
    vector_idle_a = 1'b0; vector_idle_b = 1'b1; systolic_array_idle = 1'b1;
    instruction     = make_move(24'h000500, 24'h000001, 16'd12);
    move_route_ok   = 1'b0;
    move_saw_wb_src = 1'b0;
    move_saw_win    = 1'b0;
    cap_move_pool   = FUNCT_MAX; // seed wrong; must become NOOP
    cap_move_len    = 16'd0;
    vsa_count       = 0;

    @(posedge clk); #1; controller_start = 1'b1;
    @(posedge clk); #1; controller_start = 1'b0;
    @(posedge clk); #1; // CLEAR
    @(posedge clk); #1; // EXEC_VP_START — sample execute decode
    if ((vect_source_a === VSRC_MEM) && (vect_dest_a === VDST_MEM) &&
        (mem_source_address_a === 24'h000500) &&
        (mem_dest_address_a === 24'h000001))
        move_route_ok = 1'b1;
    cap_move_pool = pooler_funct;
    cap_move_len  = length_a;
    vector_idle_a = 1'b1; // release
    while (!controller_idle)
    begin
        @(posedge clk); #1;
        if (vect_source_a === VSRC_SA_OUT || vect_source_a === VSRC_VEC_BUF)
            move_saw_wb_src = 1'b1;
        if (vect_source_a === VSRC_MEM_WIN)
            move_saw_win = 1'b1;
    end
    check((move_route_ok === 1'b1) && (cap_move_pool === FUNCT_NOOP) &&
          (cap_move_len === 16'd12) && (move_saw_wb_src === 1'b0) &&
          (move_saw_win === 1'b0) && (vsa_count === 1), 18);

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
