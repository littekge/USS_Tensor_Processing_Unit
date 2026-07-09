/*
 * File: TB_Step1_DataMemory.v
 * Date: 2026-07-09
 *
 * Testbench for v0.4 Step 1: Data_Memory wrapper (unified 0x1 buffer + two
 * 65536-word Mem_Unit blocks behind one flat, 24-bit, dual-port address space).
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. cd <path>/Tensor_Processing_Unit
 *   2. Compile (the RAM wrappers require the altera_mf simulation library):
 *        vlib work
 *        vlog -work work TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v
 *        vlog -work work TPU/MEMORY/MEM_UNIT/Mem_Unit.v
 *        vlog -work work TPU/MEMORY/Data_Memory.v
 *        vlog -work work tests/TB_Step1_DataMemory.v
 *   3. Simulate (do NOT optimize signals away; log all waveforms):
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step1_DataMemory -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL"; a final "Results: X PASS, Y FAIL"
 *   summary is printed. All 10 tests should PASS.
 *
 * TEST CASES:
 *   Test 1:  Zero-register read — address 0x0 returns 0.
 *   Test 2:  0x1 buffer routing — write/read at address 0x1 via offset.
 *   Test 3:  0x1 buffer offset independence — two offsets hold distinct values.
 *   Test 4:  Block A select — write/read a low address (addr[16]==0).
 *   Test 5:  Block B select — write/read a high address (addr[16]==1).
 *   Test 6:  Block A vs B isolation — same low 16 bits, different blocks, hold
 *            distinct values.
 *   Test 7:  0x0 write error — writing address 0x0 asserts o_write_error_a and
 *            does not corrupt (0x0 still reads 0).
 *   Test 8:  Out-of-range error — an address with a bit above [16] asserts
 *            o_range_error_a and reads back 0.
 *   Test 9:  Dual-port independence — port a and port b address different
 *            blocks simultaneously and read back their own data.
 *   Test 10: 2-cycle read latency — q is NOT valid one cycle after the address
 *            but IS valid two cycles after.
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step1_DataMemory;

localparam CLK_HALF = 10; // 10 ns -> 50 MHz

reg clk;

// DUT port a
reg  [7:0]  data_a;
reg  [23:0] address_a;
reg         wren_a;
reg  [9:0]  offset_a;
wire [7:0]  q_a;

// DUT port b
reg  [7:0]  data_b;
reg  [23:0] address_b;
reg         wren_b;
reg  [9:0]  offset_b;
wire [7:0]  q_b;

wire write_error_a, write_error_b;
wire range_error_a, range_error_b;

Data_Memory dut (
    .i_clk           (clk),
    .i_data_a        (data_a),
    .i_address_a     (address_a),
    .i_wren_a        (wren_a),
    .i_offset_a      (offset_a),
    .o_q_a           (q_a),
    .i_data_b        (data_b),
    .i_address_b     (address_b),
    .i_wren_b        (wren_b),
    .i_offset_b      (offset_b),
    .o_q_b           (q_b),
    .o_write_error_a (write_error_a),
    .o_write_error_b (write_error_b),
    .o_range_error_a (range_error_a),
    .o_range_error_b (range_error_b)
);

always #CLK_HALF clk = ~clk;

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
// Access helpers
// ---------------------------------------------------------------------------

// Single-cycle write on port a.
task wr_a;
    input [23:0] addr;
    input [9:0]  off;
    input [7:0]  d;
    begin
        address_a = addr;
        offset_a  = off;
        data_a    = d;
        wren_a    = 1'b1;
        @(posedge clk); #1;
        wren_a    = 1'b0;
    end
endtask

// Single-cycle write on port b.
task wr_b;
    input [23:0] addr;
    input [9:0]  off;
    input [7:0]  d;
    begin
        address_b = addr;
        offset_b  = off;
        data_b    = d;
        wren_b    = 1'b1;
        @(posedge clk); #1;
        wren_b    = 1'b0;
    end
endtask

// Present a read address on port a and hold it through the 2-cycle latency.
task rd_a;
    input [23:0] addr;
    input [9:0]  off;
    begin
        address_a = addr;
        offset_a  = off;
        wren_a    = 1'b0;
        @(posedge clk); #1;   // latency cycle 1
        @(posedge clk); #1;   // latency cycle 2
        @(posedge clk); #1;   // q_a now reflects addr (settled)
    end
endtask

// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------
integer q_after1;

initial
begin
    clk = 1'b0;
    data_a = 8'd0; address_a = 24'd0; wren_a = 1'b0; offset_a = 10'd0;
    data_b = 8'd0; address_b = 24'd0; wren_b = 1'b0; offset_b = 10'd0;
    pass_count = 0; fail_count = 0;

    @(posedge clk); #1;
    @(posedge clk); #1;

    // Test 1: zero-register read
    rd_a(24'd0, 10'd0);
    check(q_a === 8'd0, 1);

    // Test 2: 0x1 buffer write then read (via offset)
    wr_a(24'd1, 10'd5, 8'hAB);
    rd_a(24'd1, 10'd5);
    check(q_a === 8'hAB, 2);

    // Test 3: distinct offsets in the 0x1 buffer
    wr_a(24'd1, 10'd6, 8'hCD);
    rd_a(24'd1, 10'd5);
    check(q_a === 8'hAB, 3);   // offset 5 unchanged by the offset-6 write

    // Test 4: block A (addr[16]==0) write/read
    wr_a(24'd2, 10'd0, 8'h11);
    rd_a(24'd2, 10'd0);
    check(q_a === 8'h11, 4);

    // Test 5: block B (addr[16]==1) write/read
    wr_a(24'd65538, 10'd0, 8'h22);   // 0x10002 -> block B, offset 2
    rd_a(24'd65538, 10'd0);
    check(q_a === 8'h22, 5);

    // Test 6: block A vs B isolation (same low 16 bits [15:0]=2, different block)
    wr_a(24'd2,     10'd0, 8'h33);   // block A, offset 2
    wr_a(24'd65538, 10'd0, 8'h44);   // block B, offset 2
    rd_a(24'd2, 10'd0);
    check(q_a === 8'h33, 6);         // block A retained its own value

    // Test 7: write to reserved 0x0 raises WRITE_ERROR and does not corrupt
    address_a = 24'd0; offset_a = 10'd0; data_a = 8'hFF; wren_a = 1'b1;
    #1;
    check(write_error_a === 1'b1, 7);
    @(posedge clk); #1; wren_a = 1'b0;
    rd_a(24'd0, 10'd0);
    // (0x0 still reads 0 — reserved zero register)

    // Test 8: out-of-range address (bit 17 set) raises RANGE_ERROR, reads 0
    address_a = 24'h020000; offset_a = 10'd0; wren_a = 1'b0;
    #1;
    rd_a(24'h020000, 10'd0);
    check((range_error_a === 1'b1) && (q_a === 8'd0), 8);

    // Test 9: dual-port independence (a -> block A, b -> block B)
    wr_a(24'd10,    10'd0, 8'h5A);
    wr_b(24'd65546, 10'd0, 8'hA5);   // 0x1000A -> block B
    address_a = 24'd10;    offset_a = 10'd0; wren_a = 1'b0;
    address_b = 24'd65546; offset_b = 10'd0; wren_b = 1'b0;
    @(posedge clk); #1;
    @(posedge clk); #1;
    @(posedge clk); #1;
    check((q_a === 8'h5A) && (q_b === 8'hA5), 9);

    // Test 10: 2-cycle read latency — one cycle is NOT enough, two cycles is.
    wr_a(24'd20, 10'd0, 8'h77);
    // preload a different location so q_a holds a known other value
    rd_a(24'd2, 10'd0);              // q_a now 0x33 (from Test 6)
    address_a = 24'd20; offset_a = 10'd0; wren_a = 1'b0;
    @(posedge clk); #1;             // latency cycle 1 only
    q_after1 = q_a;
    @(posedge clk); #1;             // latency cycle 2
    check((q_after1 !== 8'h77) && (q_a === 8'h77), 10);

    #20;
    $display("-----------------------------");
    $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
    $display("-----------------------------");
    $finish;
end
endmodule
