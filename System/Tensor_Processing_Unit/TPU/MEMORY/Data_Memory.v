/*
 * File: Data_Memory.v
 * Author: Gabe Litteken
 * Date: 2026-07-09
 *
 * Unified data-memory wrapper. Abstracts the TPU_0x1_Buffer (I/O staging) and
 * two 65536-word Mem_Unit blocks (A, B) into a single flat, 24-bit, dual-port
 * (a/b) address space compliant with the Functional TPU ISA v0.4.
 *
 * Per-port address decode (identical for ports a and b):
 *   0x0                -> hardwired-zero register (read returns 0; a write
 *                         raises WRITE_ERROR and is discarded)
 *   0x1                -> TPU_0x1_Buffer, internally addressed by the offset
 *                         input (the buffer is architecturally isolated: a
 *                         tensor at 0x1 lives at buffer offsets 0,1,2,...)
 *   0x2 .. 0x1FFFF     -> Mem_Unit. address[16] selects block B (1) / block A
 *                         (0); address[15:0] is the within-block offset
 *   >= 0x20000         -> out of range (any bit above [16] set); read returns
 *                         0 and o_range_error is asserted
 *
 * Both Mem_Unit blocks and the buffer are always presented an address on both
 * ports; only the write enable is gated by the region decode. The output data
 * mux selects among {zero, buffer, block A, block B} using the region decode
 * delayed by two clock cycles so it aligns with the RAMs' 2-cycle registered
 * read latency (registered address + registered output). This preserves the
 * 2-cycle latency the rest of the datapath expects.
 */
module Data_Memory (

    input              i_clk,

    // Port a (programmer while programming; vector processor a while running)
    input      [7:0]   i_data_a,
    input      [23:0]  i_address_a,
    input              i_wren_a,
    input      [9:0]   i_offset_a,     // 0x1 buffer internal offset (port a)
    output wire [7:0]  o_q_a,

    // Port b (vector processor b)
    input      [7:0]   i_data_b,
    input      [23:0]  i_address_b,
    input              i_wren_b,
    input      [9:0]   i_offset_b,     // 0x1 buffer internal offset (port b)
    output wire [7:0]  o_q_b,

    // Error flags (combinational; observability for callers / testbenches)
    output wire        o_write_error_a,  // write attempted at 0x0 (port a)
    output wire        o_write_error_b,  // write attempted at 0x0 (port b)
    output wire        o_range_error_a,  // access above the implemented range (a)
    output wire        o_range_error_b   // access above the implemented range (b)

    // ---------- PARAMETERS ---------- //
    // ---------- END PARAMETERS ---------- //

    // ---------- DEBUG ---------- //
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //

// Region select codes for the output data mux.
parameter [2:0]
    REGION_ZERO  = 3'd0,   // reserved 0x0: hardwired zero
    REGION_BUF   = 3'd1,   // reserved 0x1: TPU_0x1_Buffer
    REGION_MEM_A = 3'd2,   // Mem_Unit block A (address[16] == 0)
    REGION_MEM_B = 3'd3,   // Mem_Unit block B (address[16] == 1)
    REGION_RANGE = 3'd4;   // out of range (address bit above [16] set)

// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// ----- Port a region decode (combinational) -----
wire a_out_of_range = |i_address_a[23:17];
wire a_is_zero      = (i_address_a == 24'd0);
wire a_is_buf       = (i_address_a == 24'd1);
wire a_sel_block_b  = i_address_a[16];

reg [2:0] region_a;
always @(*)
begin
    if (a_out_of_range)
        region_a = REGION_RANGE;
    else if (a_is_zero)
        region_a = REGION_ZERO;
    else if (a_is_buf)
        region_a = REGION_BUF;
    else if (a_sel_block_b)
        region_a = REGION_MEM_B;
    else
        region_a = REGION_MEM_A;
end

// ----- Port b region decode (combinational) -----
wire b_out_of_range = |i_address_b[23:17];
wire b_is_zero      = (i_address_b == 24'd0);
wire b_is_buf       = (i_address_b == 24'd1);
wire b_sel_block_b  = i_address_b[16];

reg [2:0] region_b;
always @(*)
begin
    if (b_out_of_range)
        region_b = REGION_RANGE;
    else if (b_is_zero)
        region_b = REGION_ZERO;
    else if (b_is_buf)
        region_b = REGION_BUF;
    else if (b_sel_block_b)
        region_b = REGION_MEM_B;
    else
        region_b = REGION_MEM_A;
end

// ----- Per-target write enables (gated by region; writes to 0x0 / out-of-range
//       never reach a RAM) -----
wire buf_wren_a  = i_wren_a & (region_a == REGION_BUF);
wire buf_wren_b  = i_wren_b & (region_b == REGION_BUF);
wire memA_wren_a = i_wren_a & (region_a == REGION_MEM_A);
wire memA_wren_b = i_wren_b & (region_b == REGION_MEM_A);
wire memB_wren_a = i_wren_a & (region_a == REGION_MEM_B);
wire memB_wren_b = i_wren_b & (region_b == REGION_MEM_B);

// ----- Error flags -----
assign o_write_error_a = i_wren_a & a_is_zero;
assign o_write_error_b = i_wren_b & b_is_zero;
assign o_range_error_a = a_out_of_range;
assign o_range_error_b = b_out_of_range;

// ----- Submodule read data -----
wire [7:0] buf_q_a,  buf_q_b;
wire [7:0] memA_q_a, memA_q_b;
wire [7:0] memB_q_a, memB_q_b;

TPU_0x1_Buffer buff (
    .clock    (i_clk),
    .data_a   (i_data_a),
    .address_a(i_offset_a),
    .wren_a   (buf_wren_a),
    .data_b   (i_data_b),
    .address_b(i_offset_b),
    .wren_b   (buf_wren_b),
    .q_a      (buf_q_a),
    .q_b      (buf_q_b)
);

Mem_Unit mem_a (
    .clock    (i_clk),
    .data_a   (i_data_a),
    .address_a(i_address_a[15:0]),
    .wren_a   (memA_wren_a),
    .data_b   (i_data_b),
    .address_b(i_address_b[15:0]),
    .wren_b   (memA_wren_b),
    .q_a      (memA_q_a),
    .q_b      (memA_q_b)
);

Mem_Unit mem_b (
    .clock    (i_clk),
    .data_a   (i_data_a),
    .address_a(i_address_a[15:0]),
    .wren_a   (memB_wren_a),
    .data_b   (i_data_b),
    .address_b(i_address_b[15:0]),
    .wren_b   (memB_wren_b),
    .q_a      (memB_q_a),
    .q_b      (memB_q_b)
);

// ----- Region-select pipeline: the RAMs have a 2-cycle registered read
//       latency, so the region decode driving the output mux must be delayed
//       by two cycles to line up with q on each RAM. -----
reg [2:0] region_a_d1, region_a_d2;
reg [2:0] region_b_d1, region_b_d2;
always @(posedge i_clk)
begin
    region_a_d1 <= region_a;
    region_a_d2 <= region_a_d1;
    region_b_d1 <= region_b;
    region_b_d2 <= region_b_d1;
end

// ----- Output data mux (aligned with the 2-cycle read latency) -----
assign o_q_a = (region_a_d2 == REGION_ZERO)  ? 8'd0     :
               (region_a_d2 == REGION_BUF)   ? buf_q_a  :
               (region_a_d2 == REGION_MEM_A) ? memA_q_a :
               (region_a_d2 == REGION_MEM_B) ? memB_q_a :
               8'd0;   // REGION_RANGE

assign o_q_b = (region_b_d2 == REGION_ZERO)  ? 8'd0     :
               (region_b_d2 == REGION_BUF)   ? buf_q_b  :
               (region_b_d2 == REGION_MEM_A) ? memA_q_b :
               (region_b_d2 == REGION_MEM_B) ? memB_q_b :
               8'd0;   // REGION_RANGE

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
