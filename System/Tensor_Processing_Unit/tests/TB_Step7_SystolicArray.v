/*
 * File: TB_Step7_SystolicArray.v
 * Date: 2026-06-22
 *
 * Testbench for Build Step 7: Systolic_Array module (and its submodules
 * Multiply_Accumulate_Unit and Systolic_Array_Input_Buffer).
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. Open a terminal and cd to the project root:
 *        cd <path>/Tensor_Processing_Unit
 *
 *   2. Compile (scfifo wrapper requires the altera_mf simulation library):
 *        vlib work
 *        vlog -work work TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v
 *        vlog -work work TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v
 *        vlog -work work TPU/SYSTOLIC_ARRAY/Systolic_Array.v
 *        vlog -work work tests/TB_Step7_SystolicArray.v
 *
 *   3. Simulate (do NOT optimize signals away; log all waveforms):
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step7_SystolicArray -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary prints total pass/fail counts.
 *   All 7 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: Reset state — idle HIGH, ramp counter and rdreq cleared.
 *   Test 2: Clear flushes all input buffers (all_empty HIGH after clear).
 *   Test 3: systolic_array_start lowers idle (FSM enters LOAD).
 *   Test 4: Read-request skew — buffer M is enabled on the Mth LOAD cycle
 *           (top_rdreq ramps 0x01 -> 0x03 on consecutive cycles).
 *   Test 5: Read requests are gated by empty (an empty buffer never reads).
 *   Test 6: Idle returns HIGH once every buffer has drained.
 *   Test 7: A loaded MAC tile accumulates a non-zero product and the o_c
 *           output mux selects MAC(col=i_col, row=i_row); untouched tiles
 *           read back zero.
 *   Test 8: clear zeroes a MAC accumulator (residual data flushed between ops).
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step7_SystolicArray;

localparam N = 8;

// ---------------------------------------------------------------------------
// Clock / reset
// ---------------------------------------------------------------------------
localparam CLK_HALF = 10; // 10 ns -> 50 MHz

reg clk, trst;

always #CLK_HALF clk = ~clk;

task do_reset;
begin
    trst = 1'b0;
    @(posedge clk); #1;
    @(posedge clk); #1;
    trst = 1'b1;
    @(posedge clk); #1;   // START -> IDLE
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
// DUT signals
// ---------------------------------------------------------------------------
reg          clear;
reg          sa_start;
reg  [7:0]   top_data;
reg  [N-1:0] top_wrreq;
reg  [7:0]   left_data;
reg  [N-1:0] left_wrreq;
reg  [3:0]   sel_row;
reg  [3:0]   sel_col;
wire         sa_idle;
wire [31:0]  c_out;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
Systolic_Array #(.N(N)) dut (
    .i_clk                  (clk),
    .i_trst                 (trst),
    .i_clear                (clear),
    .i_systolic_array_start (sa_start),
    .o_systolic_array_idle  (sa_idle),
    .i_top_data             (top_data),
    .i_top_wrreq            (top_wrreq),
    .i_left_data            (left_data),
    .i_left_wrreq           (left_wrreq),
    .i_row                  (sel_row),
    .i_col                  (sel_col),
    .o_c                    (c_out)
);

// ---------------------------------------------------------------------------
// Helper tasks
// ---------------------------------------------------------------------------

// Push one element into every top buffer simultaneously (shared data bus).
task push_all_top;
    input [7:0] d;
    begin
        top_data  = d;
        top_wrreq = {N{1'b1}};
        @(posedge clk); #1;
        top_wrreq = {N{1'b0}};
    end
endtask

// Push one element into every left buffer simultaneously.
task push_all_left;
    input [7:0] d;
    begin
        left_data  = d;
        left_wrreq = {N{1'b1}};
        @(posedge clk); #1;
        left_wrreq = {N{1'b0}};
    end
endtask

// Push one element into a single top buffer (one-hot wrreq).
task push_top;
    input integer idx;
    input [7:0]   d;
    begin
        top_data  = d;
        top_wrreq = ({{(N-1){1'b0}}, 1'b1} << idx);
        @(posedge clk); #1;
        top_wrreq = {N{1'b0}};
    end
endtask

// Push one element into a single left buffer (one-hot wrreq).
task push_left;
    input integer idx;
    input [7:0]   d;
    begin
        left_data  = d;
        left_wrreq = ({{(N-1){1'b0}}, 1'b1} << idx);
        @(posedge clk); #1;
        left_wrreq = {N{1'b0}};
    end
endtask

// Pulse the start signal for exactly one clock cycle.
task pulse_start;
    begin
        sa_start = 1'b1;
        @(posedge clk); #1;
        sa_start = 1'b0;
    end
endtask

// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------
integer guard;

initial
begin
    clk        = 1'b0;
    trst       = 1'b1;
    clear      = 1'b0;
    sa_start   = 1'b0;
    top_data   = 8'd0;
    top_wrreq  = {N{1'b0}};
    left_data  = 8'd0;
    left_wrreq = {N{1'b0}};
    sel_row    = 4'd0;
    sel_col    = 4'd0;
    pass_count = 0;
    fail_count = 0;

    // -----------------------------------------------------------------------
    // Test 1: Reset state — idle HIGH, ramp/rdreq cleared.
    // -----------------------------------------------------------------------
    do_reset;
    check((sa_idle === 1'b1) && (dut.load_cnt === 8'd0) &&
          (dut.top_rdreq === {N{1'b0}}) && (dut.left_rdreq === {N{1'b0}}), 1);

    // -----------------------------------------------------------------------
    // Test 2: Clear flushes all input buffers.
    // -----------------------------------------------------------------------
    do_reset;
    push_all_top(8'd9);
    push_all_left(8'd9);
    @(posedge clk); #1;
    // Buffers now hold data -> all_empty must be LOW.
    if (dut.all_empty !== 1'b0)
    begin
        check(1'b0, 2); // buffers did not accept data
    end
    else
    begin
        clear = 1'b1;
        @(posedge clk); #1;
        clear = 1'b0;
        // Allow scfifo sclr to take effect.
        repeat (3) @(posedge clk);
        #1;
        check(dut.all_empty === 1'b1, 2);
    end

    // -----------------------------------------------------------------------
    // Test 3: start lowers idle (FSM enters LOAD).
    // -----------------------------------------------------------------------
    do_reset;
    push_all_top(8'd4);
    push_all_left(8'd4);
    pulse_start;          // IDLE -> LOAD on this edge
    check((sa_idle === 1'b0) && (dut.S === dut.LOAD), 3);

    // -----------------------------------------------------------------------
    // Test 4: Read-request skew. On the first LOAD cycle only buffer 0 reads;
    //         on the second, buffers 0 and 1 read (diagonal offset).
    // -----------------------------------------------------------------------
    do_reset;
    // Load every buffer with several elements so none drain during the check.
    repeat (4) push_all_top(8'd2);
    repeat (4) push_all_left(8'd2);
    pulse_start;          // now in LOAD, load_cnt == 0
    // First LOAD cycle already active after pulse_start settle.
    if (dut.top_rdreq !== 8'h01)
    begin
        check(1'b0, 4);
    end
    else
    begin
        @(posedge clk); #1;   // second LOAD cycle, load_cnt == 1
        check(dut.top_rdreq === 8'h03, 4);
    end

    // -----------------------------------------------------------------------
    // Test 5: Read requests are gated by empty. Only buffer 0 is loaded, so
    //         buffer 1's rdreq must never assert even after the ramp passes it.
    // -----------------------------------------------------------------------
    do_reset;
    push_top(0, 8'd7);
    push_left(0, 8'd7);
    pulse_start;
    begin : empty_gate_check
        reg bit1_seen;
        bit1_seen = 1'b0;
        guard = 0;
        while ((sa_idle === 1'b0) && (guard < 64))
        begin
            if (dut.top_rdreq[1] === 1'b1)  bit1_seen = 1'b1;
            if (dut.left_rdreq[1] === 1'b1) bit1_seen = 1'b1;
            @(posedge clk); #1;
            guard = guard + 1;
        end
        check(bit1_seen === 1'b0, 5);
    end

    // -----------------------------------------------------------------------
    // Test 6: Idle returns HIGH once every buffer has drained.
    // -----------------------------------------------------------------------
    do_reset;
    repeat (2) push_all_top(8'd1);
    repeat (2) push_all_left(8'd1);
    pulse_start;
    guard = 0;
    while ((sa_idle === 1'b0) && (guard < 128))
    begin
        @(posedge clk); #1;
        guard = guard + 1;
    end
    check((sa_idle === 1'b1) && (guard < 128), 6);

    // -----------------------------------------------------------------------
    // Test 7: A loaded MAC tile accumulates a non-zero product, and the o_c
    //         output mux selects MAC(col=i_col, row=i_row). Only top_buf[0]
    //         and left_buf[0] are loaded, so only MAC(0,0) sees real operands.
    // -----------------------------------------------------------------------
    do_reset;
    push_top(0, 8'd2);
    push_top(0, 8'd2);
    push_left(0, 8'd3);
    push_left(0, 8'd3);
    pulse_start;
    guard = 0;
    while ((sa_idle === 1'b0) && (guard < 128))
    begin
        @(posedge clk); #1;
        guard = guard + 1;
    end
    // Read MAC(0,0): product of 2*3 accumulated -> non-zero multiple of 6.
    sel_col = 4'd0;
    sel_row = 4'd0;
    #1;
    begin : decode_check
        reg corner_ok;
        reg other_zero;
        corner_ok = (c_out != 32'd0) && ((c_out % 32'd6) == 32'd0);
        // Read an untouched tile MAC(1,1): operands were never loaded -> zero.
        sel_col = 4'd1;
        sel_row = 4'd1;
        #1;
        other_zero = (c_out === 32'd0);
        check(corner_ok && other_zero, 7);
    end

    // -----------------------------------------------------------------------
    // Test 8: clear zeroes a MAC accumulator. Accumulate a non-zero product
    //         into MAC(0,0), then assert clear and confirm the accumulator is
    //         flushed to zero (residual data does not survive between ops).
    // -----------------------------------------------------------------------
    do_reset;
    push_top(0, 8'd2);
    push_top(0, 8'd2);
    push_left(0, 8'd3);
    push_left(0, 8'd3);
    pulse_start;
    guard = 0;
    while ((sa_idle === 1'b0) && (guard < 128))
    begin
        @(posedge clk); #1;
        guard = guard + 1;
    end
    sel_col = 4'd0;
    sel_row = 4'd0;
    #1;
    begin : clear_check
        reg nonzero_before;
        nonzero_before = (c_out != 32'd0);
        // Assert clear for one cycle to flush the MAC accumulators.
        clear = 1'b1;
        @(posedge clk); #1;
        clear = 1'b0;
        @(posedge clk); #1;
        sel_col = 4'd0;
        sel_row = 4'd0;
        #1;
        check((nonzero_before === 1'b1) && (c_out === 32'd0), 8);
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
