/*
 * File: TB_Step5_Activator.v
 * Date: 2026-06-16
 *
 * Testbench for Build Step 5: Activator module.
 * Tests combinational implementation of activation functions and control
 * logic. This is a unit test; connections to other modules are not verified.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. Open a terminal and cd to the project root:
 *        cd <path>/Tensor_Processing_Unit
 *
 *   2. Compile:
 *        vlib work
 *        vlog -work work TPU/PROCESSING/Activator.v
 *        vlog -work work tests/TB_Step5_Activator.v
 *
 *   3. Simulate:
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step5_Activator -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary prints total pass/fail counts.
 *   All N tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: o_data is forced to 45 when i_trst LOW
 *   Test 2: o_data is forced to 45 when i_clear HIGH
 *   Test 3: o_data is forced to 45 when i_enable LOW
 *   Test 4: i_enable routes to o_write for a non-NOOP function
 *   Test 5: NOOP function — placeholder o_data and o_write suppressed (LOW)
 *   Test 6: ReLu operation functions correctly
 * -------------------------------------------------------------------------
 */
 
`timescale 1 ns / 1 ps
 
module TB_Step5_Activator;
 
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

reg act_enable;
reg clear;
wire write;
reg [2:0] activator_funct;
reg signed [7:0] in_data;
wire signed [7:0] out_data;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------

Activator dut (
	.i_clk(clk),
	.i_trst(trst),
	.i_enable(act_enable),
	.i_clear(clear),
	.o_write(write),
	.i_activator_op(activator_funct),
	.i_data(in_data),
	.o_data(out_data)
);


// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------
initial
begin
	clk = 1'b0;
	act_enable = 1'b0;
	clear = 1'b0;
	trst = 1'b1;
	activator_funct = 3'd7;
	in_data = 8'd0;
	pass_count = 0;
   fail_count = 0;
	
	// -----------------------------------------------------------------------
   // Test 1: o_data is forced to 45 when i_trst LOW
   // -----------------------------------------------------------------------
	trst = 1'b0;
	act_enable = 1'b1;
	clear = 1'b0;
	in_data = 8'd20; //should be overridden if test is successful
	@(posedge clk); #1;
	@(posedge clk); #1;
	check(out_data === 8'd45, 1);
	do_reset;
	
	// -----------------------------------------------------------------------
   // Test 2: o_data is forced to 45 when i_clear HIGH
   // -----------------------------------------------------------------------
	trst = 1'b1;
	act_enable = 1'b1;
	clear = 1'b1;
	in_data = 8'd20; //should be overridden if test is successful
	@(posedge clk); #1;
	@(posedge clk); #1;
	check(out_data === 8'd45, 2);

	// -----------------------------------------------------------------------
   // Test 3: o_data is forced to 45 when i_enable LOW
   // -----------------------------------------------------------------------
	trst = 1'b1;
	act_enable = 1'b0;
	clear = 1'b0;
	in_data = 8'd20; //should be overridden if test is successful
	@(posedge clk); #1;
	@(posedge clk); #1;
	check(out_data === 8'd45, 3);
	
	// -----------------------------------------------------------------------
   // Test 4: i_enable routes to o_write for a non-NOOP function (RELU)
   // -----------------------------------------------------------------------
	begin : write_route_test
		integer write_pass_count;
		write_pass_count = 0;
		activator_funct = 3'h0; // RELU (non-NOOP)
		act_enable = 1'b1;
		@(posedge clk); #1;
		if (write === 1'b1) write_pass_count = write_pass_count + 1;
		act_enable = 1'b0;
		@(posedge clk); #1;
		if (write === 1'b0) write_pass_count = write_pass_count + 1;
		check(write_pass_count === 2, 4);
	end

	// -----------------------------------------------------------------------
   // Test 5: NOOP — placeholder o_data (98) and o_write suppressed even with
   //         i_enable HIGH (buffer writing is suppressed for NOOP per spec)
   // -----------------------------------------------------------------------
	trst = 1'b1;
	act_enable = 1'b1;
	clear = 1'b0;
	in_data = 8'd20;
	activator_funct = 3'h7;
	@(posedge clk); #1;
	@(posedge clk); #1;
	check((out_data === 8'd98) && (write === 1'b0), 5);
	
	// -----------------------------------------------------------------------
   // Test 6: ReLu operation functions correctly
   // -----------------------------------------------------------------------
	trst = 1'b1;
	act_enable = 1'b1;
	clear = 1'b0;
	begin : relu_test
		integer relu_pass_count;
		relu_pass_count = 0;
		activator_funct = 3'h0;

		//test positive integer
		in_data = 8'd30;
		@(posedge clk); #1;
		@(posedge clk); #1;
		if (out_data === 8'd30) relu_pass_count = relu_pass_count + 1;
		
		//test negative integer
		in_data = -8'd30;
		@(posedge clk); #1;
		@(posedge clk); #1;
		if (out_data === 8'd0) relu_pass_count = relu_pass_count + 1;
		
		check(relu_pass_count === 2, 6);
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