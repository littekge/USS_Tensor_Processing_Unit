/*
 * File: TB_Step6_ALU.v
 * Date: 2026-06-17
 *
 * Testbench for Build Step 6: ALU module.
 * Tests combinational implementation of ALU functions and control
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
 *        vlog -work work TPU/PROCESSING/ALU.v
 *        vlog -work work tests/TB_Step6_ALU.v
 *
 *   3. Simulate:
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step6_ALU -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary prints total pass/fail counts.
 *   All N tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: o_data is forced to 45 when i_trst LOW, i_enable LOW, or i_clear HIGH
 *   Test 2: i_enable routes to o_write for a non-NOOP operation
 *   Test 3: NOOP operation — placeholder o_data and o_write suppressed (LOW)
 *   Test 4: ADD operation functions correctly
 *   Test 5: positive clamping functions correctly
 *   Test 6: negative clamping functions correctly
 * -------------------------------------------------------------------------
 */
 
`timescale 1 ns / 1 ps
 
module TB_Step6_ALU;
 
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

reg alu_enable;
reg clear;
wire write;
reg [2:0] alu_funct;
reg signed [7:0] in_data_a;
reg signed [7:0] in_data_b;
wire signed [7:0] out_data;

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------

ALU dut (
	.i_clk(clk), //intentionally not connected
	.i_trst(trst),
	.i_enable(alu_enable),
	.i_clear(clear),
	.o_write(write),
	.i_alu_op(alu_funct), 
	.i_data_a(in_data_a),
	.i_data_b(in_data_b),
	.o_data(out_data)
);


// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------
initial
begin
	clk = 1'b0;
	trst = 1'b1;
	alu_enable = 1'b1;
	pass_count = 0;
	fail_count = 0;
	in_data_a = 8'd0;
	in_data_b = 8'd0;
	alu_funct = 3'd7;
	
	// -----------------------------------------------------------------------
   // Test 1: o_data is forced to 45 when i_trst LOW, i_enable LOW, or i_clear HIGH
   // -----------------------------------------------------------------------
	begin : control_test
		integer control_pass_count;
		control_pass_count = 0;
		trst = 1'b1;
		clear = 1'b0;
		alu_enable = 1'b1;
		
		trst = 1'b0;
		@(posedge clk); #1;
		if (out_data === 8'd45) control_pass_count = control_pass_count + 1;
		trst = 1'b1;
		
		clear = 1'b1;
		@(posedge clk); #1;
		if (out_data === 8'd45) control_pass_count = control_pass_count + 1;
		clear = 1'b0;
		
		alu_enable = 1'b0;
		@(posedge clk); #1;
		if (out_data === 8'd45) control_pass_count = control_pass_count + 1;
		alu_enable = 1'b1;
		
		check(control_pass_count === 3, 1);
	end
	
	// -----------------------------------------------------------------------
   // Test 2: i_enable routes to o_write for a non-NOOP operation (ADD)
   // -----------------------------------------------------------------------
	begin : write_route_test
		integer write_pass_count;
		write_pass_count = 0;
		alu_funct = 3'h0; // ADD (non-NOOP)
		alu_enable = 1'b1;
		@(posedge clk); #1;
		if (write === 1'b1) write_pass_count = write_pass_count + 1;
		alu_enable = 1'b0;
		@(posedge clk); #1;
		if (write === 1'b0) write_pass_count = write_pass_count + 1;
		check(write_pass_count === 2, 2);
	end

	// -----------------------------------------------------------------------
   // Test 3: NOOP — placeholder o_data (98) and o_write suppressed even with
   //         i_enable HIGH (buffer writing is suppressed for NOOP per spec)
   // -----------------------------------------------------------------------
	trst = 1'b1;
	alu_enable = 1'b1;
	clear = 1'b0;
	in_data_a = 8'd20;
	in_data_b = 8'd21;
	alu_funct = 3'h7;
	@(posedge clk); #1;
	check((out_data === 8'd98) && (write === 1'b0), 3);
	
	// -----------------------------------------------------------------------
   // Test 4: ADD operation functions correctly
   // -----------------------------------------------------------------------
	trst = 1'b1;
	alu_enable = 1'b1;
	clear = 1'b0;
	in_data_a = -8'd1;
	in_data_b = -8'd1;
	alu_funct = 3'h0;
	@(posedge clk); #1;
	check(out_data === -8'd2, 4);
	
	// -----------------------------------------------------------------------
   // Test 5: positive clamping functions correctly
   // -----------------------------------------------------------------------
	trst = 1'b1;
	alu_enable = 1'b1;
	clear = 1'b0;
	in_data_a = 8'd127;
	in_data_b = 8'd100;
	alu_funct = 3'h0;
	@(posedge clk); #1;
	check(out_data === 8'd127, 5);
	
	// -----------------------------------------------------------------------
   // Test 6: negative clamping functions correctly
   // -----------------------------------------------------------------------
	trst = 1'b1;
	alu_enable = 1'b1;
	clear = 1'b0;
	in_data_a = -8'd127;
	in_data_b = -8'd100;
	alu_funct = 3'h0;
	@(posedge clk); #1;
	check(out_data === -8'd128, 6);
	
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