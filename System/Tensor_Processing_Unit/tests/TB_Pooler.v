/*
 * File: TB_Pooler.v
 * Date: 2026-07-16
 *
 * Testbench for v0.6 Step 1: Pooler module (streaming max-pooling reducer).
 * Unit test; connections to other modules are not verified.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. cd <path>/Tensor_Processing_Unit
 *
 *   2. Compile:
 *        vlib work
 *        vlog -work work TPU/PROCESSING/Pooler.v
 *        vlog -work work tests/TB_Pooler.v
 *
 *   3. Simulate:
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Pooler -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL". A final summary prints total counts.
 *   All tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: single-element window emits the value (identity is min, not 0)
 *   Test 2: clear resets the accumulator to the reduction identity
 *   Test 3: single multi-element window — max of the stream
 *   Test 4: two windows back-to-back reset independently at each window_end
 *   Test 5: min-fill (-128) values never win the comparison
 *   Test 6: all-fill window reduces to the identity value (-128)
 *   Test 7: NO OP suppresses o_write even on window_end
 *   Test 8: o_write stays LOW mid-window (window_end LOW)
 *   Test 9: o_write stays LOW when i_enable is LOW
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Pooler;

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
// DUT signals
// ---------------------------------------------------------------------------
reg              pool_enable;
reg              clear;
reg              window_end;
reg  [2:0]       pooler_op;
reg  signed [7:0] in_data;
wire             write;
wire signed [7:0] out_data;

localparam [2:0] MAX_OP = 3'h0;
localparam [2:0] NOOP   = 3'h7;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
Pooler dut (
    .i_clk        (clk),
    .i_trst       (trst),
    .i_enable     (pool_enable),
    .i_clear      (clear),
    .i_window_end (window_end),
    .o_write      (write),
    .i_pooler_op  (pooler_op),
    .i_data       (in_data),
    .o_data       (out_data)
);

// ---------------------------------------------------------------------------
// Helper: fold one non-final window element (enable HIGH, window_end LOW).
// Advances the accumulator across one clock edge.
// ---------------------------------------------------------------------------
task fold_val;
    input signed [7:0] d;
    begin
        pool_enable = 1'b1;
        window_end  = 1'b0;
        in_data     = d;
        #1;                 // settle combinational outputs
        @(posedge clk); #1; // acc <= fold(acc, d)
    end
endtask

// ---------------------------------------------------------------------------
// Helper: one idle cycle (enable LOW) between windows to mimic the vector
// processor's inter-element gaps; the accumulator must persist.
// ---------------------------------------------------------------------------
task idle_cycle;
begin
    pool_enable = 1'b0;
    window_end  = 1'b0;
    #1;
    @(posedge clk); #1;
end
endtask

// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------
initial
begin
    clk         = 1'b0;
    trst        = 1'b1;
    pool_enable = 1'b0;
    clear       = 1'b0;
    window_end  = 1'b0;
    pooler_op   = MAX_OP;
    in_data     = 8'd0;
    pass_count  = 0;
    fail_count  = 0;

    do_reset;

    // -----------------------------------------------------------------------
    // Test 1: single-element window emits the value. acc starts at the min
    // identity (-128), so max(-128, -5) = -5 proves identity is min (not 0).
    // -----------------------------------------------------------------------
    pool_enable = 1'b1; window_end = 1'b1; in_data = -8'sd5; #1;
    check((write === 1'b1) && (out_data === -8'sd5), 1);
    @(posedge clk); #1;
    pool_enable = 1'b0; window_end = 1'b0;

    // -----------------------------------------------------------------------
    // Test 2: clear resets accumulator to identity. Fold a big value, clear,
    // then a single-element window of -20 must emit -20 (the big value gone).
    // -----------------------------------------------------------------------
    fold_val(8'sd100);
    clear = 1'b1; pool_enable = 1'b0; window_end = 1'b0; #1;
    @(posedge clk); #1;               // acc <= identity
    clear = 1'b0;
    pool_enable = 1'b1; window_end = 1'b1; in_data = -8'sd20; #1;
    check((write === 1'b1) && (out_data === -8'sd20), 2);
    @(posedge clk); #1;
    pool_enable = 1'b0; window_end = 1'b0;
    do_reset;

    // -----------------------------------------------------------------------
    // Test 3: single multi-element window {3,7,2,5} -> max 7.
    // -----------------------------------------------------------------------
    fold_val(8'sd3);
    fold_val(8'sd7);
    fold_val(8'sd2);
    pool_enable = 1'b1; window_end = 1'b1; in_data = 8'sd5; #1;
    check((write === 1'b1) && (out_data === 8'sd7), 3);
    @(posedge clk); #1;
    pool_enable = 1'b0; window_end = 1'b0;
    do_reset;

    // -----------------------------------------------------------------------
    // Test 4: two windows back-to-back reset independently.
    //   Window A {1,9,4} -> 9 ; Window B {2,3} -> 3 (NOT 9 again).
    // -----------------------------------------------------------------------
    begin : multi_window
        integer win_pass;
        win_pass = 0;
        // Window A
        fold_val(8'sd1);
        fold_val(8'sd9);
        pool_enable = 1'b1; window_end = 1'b1; in_data = 8'sd4; #1;
        if ((write === 1'b1) && (out_data === 8'sd9)) win_pass = win_pass + 1;
        @(posedge clk); #1;                 // window_end -> acc resets to identity
        // Window B (back-to-back, no idle)
        pool_enable = 1'b1; window_end = 1'b0; in_data = 8'sd2; #1;
        @(posedge clk); #1;
        pool_enable = 1'b1; window_end = 1'b1; in_data = 8'sd3; #1;
        if ((write === 1'b1) && (out_data === 8'sd3)) win_pass = win_pass + 1;
        @(posedge clk); #1;
        pool_enable = 1'b0; window_end = 1'b0;
        check(win_pass === 2, 4);
    end
    do_reset;

    // -----------------------------------------------------------------------
    // Test 5: min-fill (-128) values never win. {-128,-3,-128,-50} -> -3.
    // Uses idle cycles between elements to confirm the accumulator persists.
    // -----------------------------------------------------------------------
    fold_val(-8'sd128);
    idle_cycle;
    fold_val(-8'sd3);
    idle_cycle;
    fold_val(-8'sd128);
    idle_cycle;
    pool_enable = 1'b1; window_end = 1'b1; in_data = -8'sd50; #1;
    check((write === 1'b1) && (out_data === -8'sd3), 5);
    @(posedge clk); #1;
    pool_enable = 1'b0; window_end = 1'b0;
    do_reset;

    // -----------------------------------------------------------------------
    // Test 6: all-fill window {-128,-128} -> -128 (legitimate identity result).
    // -----------------------------------------------------------------------
    fold_val(-8'sd128);
    pool_enable = 1'b1; window_end = 1'b1; in_data = -8'sd128; #1;
    check((write === 1'b1) && (out_data === -8'sd128), 6);
    @(posedge clk); #1;
    pool_enable = 1'b0; window_end = 1'b0;
    do_reset;

    // -----------------------------------------------------------------------
    // Test 7: NO OP suppresses o_write even on window_end.
    // -----------------------------------------------------------------------
    pooler_op = NOOP;
    pool_enable = 1'b1; window_end = 1'b1; in_data = 8'sd42; #1;
    check(write === 1'b0, 7);
    @(posedge clk); #1;
    pool_enable = 1'b0; window_end = 1'b0;
    pooler_op = MAX_OP;
    do_reset;

    // -----------------------------------------------------------------------
    // Test 8: o_write stays LOW mid-window (enable HIGH, window_end LOW).
    // -----------------------------------------------------------------------
    pool_enable = 1'b1; window_end = 1'b0; in_data = 8'sd11; #1;
    check(write === 1'b0, 8);
    @(posedge clk); #1;
    pool_enable = 1'b0; window_end = 1'b0;

    // -----------------------------------------------------------------------
    // Test 9: o_write stays LOW when i_enable is LOW even if window_end HIGH.
    // -----------------------------------------------------------------------
    pool_enable = 1'b0; window_end = 1'b1; in_data = 8'sd11; #1;
    check(write === 1'b0, 9);
    @(posedge clk); #1;
    window_end = 1'b0;

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
