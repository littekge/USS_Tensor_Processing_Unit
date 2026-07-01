/*
 * File: TB_Step4_VectorProcessor.v
 * Date: 2026-06-15
 *
 * Testbench for Build Step 4: Vector_Processor module.
 * Uses behavioral RAM and FIFO models to exercise the Vector_Processor
 * in isolation from the rest of the TPU.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. Open a terminal and cd to the project root:
 *        cd <path>/Tensor_Processing_Unit
 *
 *   2. Compile:
 *        vlib work
 *        vlog -work work TPU/PROCESSING/Vector_Processor.v
 *        vlog -work work tests/TB_Step4_VectorProcessor.v
 *
 *   3. Simulate:
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step4_VectorProcessor \
 *             -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary prints total pass/fail counts.
 *   All 10 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1:  After reset — o_vector_idle is HIGH immediately
 *   Test 2:  VSRC_MEM→VDST_ACT — element_valid pulses 4 times with correct data
 *   Test 3:  VSRC_MEM→VDST_SA_A — rows of matrix pushed to correct top SA buffers
 *   Test 4:  VSRC_MEM→VDST_SA_B — columns loaded via stride access into left SA buffers
 *   Test 5:  VSRC_VEC_BUF→VDST_MEM — VB data read and written to weight memory
 *   Test 6:  VSRC_SA_OUT→VDST_MEM — SA accumulator requantized and written; saturation check
 *   Test 7:  VSRC_MEM→VDST_MEM with dest=0x0000 — VP enters MEM_ERROR and stalls
 *   Test 8:  VSRC_MEM (src=0x0001) → VDST_ACT — 0x1 Buffer as source
 *   Test 9:  VSRC_VEC_BUF → VDST_MEM (dest=0x0001) — 0x1 Buffer as destination
 *   Test 10: VSRC_MEM (src=0x0000) → VDST_ACT — zero source always returns 0x00
 *   Test 11: VSRC_MEM → VDST_ALU_A — element_valid streaming to the ALU input
 *   Test 12: VSRC_SA_OUT → VDST_MEM — negative requantization saturates to 0x80
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step4_VectorProcessor;

// ---------------------------------------------------------------------------
// Clock / reset
// ---------------------------------------------------------------------------
localparam CLK_HALF = 10;
reg clk, trst;
integer pass_count, fail_count;

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

task check;
    input            ok;
    input [127:0]    num;
    begin
        if (ok) begin
            $display("Test %0d: PASS", num);
            pass_count = pass_count + 1;
        end else begin
            $display("Test %0d: FAIL", num);
            fail_count = fail_count + 1;
        end
    end
endtask

// ---------------------------------------------------------------------------
// DUT interface
// ---------------------------------------------------------------------------
reg        vp_vector_start;
reg  [2:0] vp_vect_source;
reg  [2:0] vp_vect_dest;
reg [15:0] vp_mem_source_address;
reg [15:0] vp_mem_dest_address;
reg [15:0] vp_length;
reg  [3:0] vp_dim0;
reg  [3:0] vp_dim1;

wire        vp_vector_idle;
wire        vp_element_valid;
wire [15:0] vp_wm_address;
wire        vp_wm_wren;
wire  [7:0] vp_wm_data;
wire [15:0] vp_buf_address;
wire        vp_buf_wren;
wire  [7:0] vp_buf_data;
wire        vp_vb_rdreq;
wire  [7:0] vp_data;
wire  [7:0] vp_sa_top_data;
wire  [7:0] vp_sa_top_wrreq;
wire  [7:0] vp_sa_left_data;
wire  [7:0] vp_sa_left_wrreq;
wire  [3:0] vp_sa_row;
wire  [3:0] vp_sa_col;

// Source/destination codes matching Vector_Processor.v
localparam VSRC_MEM     = 3'd0;
localparam VSRC_SA_OUT  = 3'd1;
localparam VSRC_VEC_BUF = 3'd2;
localparam VDST_MEM     = 3'd0;
localparam VDST_SA_A    = 3'd1;
localparam VDST_SA_B    = 3'd2;
localparam VDST_ACT     = 3'd3;
localparam VDST_ALU_A   = 3'd4;
localparam VDST_ALU_B   = 3'd5;

// ---------------------------------------------------------------------------
// Behavioral memories
// ---------------------------------------------------------------------------
reg [7:0] wm_mem  [0:63];  // Weight_Memory model (64 words)
reg [7:0] buf_mem [0:63];  // 0x1 Buffer model (64 words)

reg [7:0] wm_q, buf_q;

// 1-cycle registered output RAM (write-first not needed; wren is only on write)
always @(posedge clk) begin
    if (vp_wm_wren)  wm_mem[vp_wm_address[5:0]]  <= vp_wm_data;
    if (vp_buf_wren) buf_mem[vp_buf_address[5:0]] <= vp_buf_data;
    wm_q  <= wm_mem[vp_wm_address[5:0]];
    buf_q <= buf_mem[vp_buf_address[5:0]];
end

// ---------------------------------------------------------------------------
// Behavioral Vector_Buffer (normal FIFO: rdreq → data stable next cycle)
// ---------------------------------------------------------------------------
reg [7:0] vb_fifo [0:15];
reg [3:0] vb_head, vb_tail;
reg [4:0] vb_cnt;
reg  [7:0] vb_q_reg;
wire [7:0] vb_q;

always @(posedge clk) begin
    if (vp_vb_rdreq && vb_cnt > 0) begin
        vb_q_reg <= vb_fifo[vb_head];
        vb_head  <= vb_head + 4'd1;
        vb_cnt   <= vb_cnt  - 5'd1;
    end
end

assign vb_q = vb_q_reg;

// Helper: push data into the behavioral VB from the test (blocking)
task vb_push;
    input [7:0] d;
    begin
        vb_fifo[vb_tail] = d;
        vb_tail = vb_tail + 4'd1;
        vb_cnt  = vb_cnt  + 5'd1;
    end
endtask

// ---------------------------------------------------------------------------
// SA accumulator stub: driven by module-level always block so it is always
// combinationally updated from VP's row/col selectors.
// ---------------------------------------------------------------------------
reg [31:0] sa_c_stub;

// neg_mode (Test 12): drives MAC(0,0) to a large negative accumulator so the
// requantizer must saturate to -128 (0x80).
reg neg_mode;

// Selects specific values used in Test 6:
//   MAC(row=0,col=0) → 256  → requantized to 0x01
//   MAC(row=1,col=0) → 40000 → saturates to  0x7F
// and in Test 12 (neg_mode): MAC(0,0) → -40000 → saturates to 0x80.
always @(*) begin
    if      (neg_mode && vp_sa_row == 4'd0 && vp_sa_col == 4'd0) sa_c_stub = -32'sd40000;
    else if (vp_sa_row == 4'd0 && vp_sa_col == 4'd0) sa_c_stub = 32'd256;
    else if (vp_sa_row == 4'd1 && vp_sa_col == 4'd0) sa_c_stub = 32'd40000;
    else                                               sa_c_stub = 32'd0;
end

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
Vector_Processor dut (
    .i_clk                (clk),
    .i_trst               (trst),
    .i_vector_start       (vp_vector_start),
    .i_vect_source        (vp_vect_source),
    .i_vect_dest          (vp_vect_dest),
    .i_mem_source_address (vp_mem_source_address),
    .i_mem_dest_address   (vp_mem_dest_address),
    .i_length             (vp_length),
    .i_dim0               (vp_dim0),
    .i_dim1               (vp_dim1),
    .o_vector_idle        (vp_vector_idle),
    .o_element_valid      (vp_element_valid),
    .o_wm_address         (vp_wm_address),
    .o_wm_wren            (vp_wm_wren),
    .o_wm_data            (vp_wm_data),
    .i_wm_q               (wm_q),
    .o_buf_address        (vp_buf_address),
    .o_buf_wren           (vp_buf_wren),
    .o_buf_data           (vp_buf_data),
    .i_buf_q              (buf_q),
    .o_vb_rdreq           (vp_vb_rdreq),
    .i_vb_q               (vb_q),
    .o_data               (vp_data),
    .o_sa_top_data        (vp_sa_top_data),
    .o_sa_top_wrreq       (vp_sa_top_wrreq),
    .o_sa_left_data       (vp_sa_left_data),
    .o_sa_left_wrreq      (vp_sa_left_wrreq),
    .o_sa_row             (vp_sa_row),
    .o_sa_col             (vp_sa_col),
    .i_sa_c               (sa_c_stub)
);

// Pin the requantization scale so Tests 6/12 (SA-output requant + saturation)
// verify the datapath deterministically, independent of the file's
// REQUANT_SHIFT default (which is tuned per neural network).
defparam dut.REQUANT_SHIFT = 8;

// ---------------------------------------------------------------------------
// Signal capture registers (updated every clock by always block)
// ---------------------------------------------------------------------------
// element_valid event log (Test 2)
integer ev_count;
reg     ev_ok;
reg [7:0] ev_data [0:7];

// SA wrreq event log (Tests 3 & 4)
integer sa_cnt;
reg     sa_ok;
reg [7:0] sa_wrreq_log [0:7];
reg [7:0] sa_data_log  [0:7];
reg       capture_sa_top;   // 1 → log top wrreq; 0 → log left wrreq
reg       capture_active;   // enable capture

always @(posedge clk) begin
    if (vp_element_valid && ev_count < 8) begin
        ev_data[ev_count] <= vp_data;
        ev_count          <= ev_count + 1;
    end
    if (capture_active) begin
        if (capture_sa_top && vp_sa_top_wrreq != 8'd0 && sa_cnt < 8) begin
            sa_wrreq_log[sa_cnt] <= vp_sa_top_wrreq;
            sa_data_log[sa_cnt]  <= vp_sa_top_data;
            sa_cnt               <= sa_cnt + 1;
        end
        if (!capture_sa_top && vp_sa_left_wrreq != 8'd0 && sa_cnt < 8) begin
            sa_wrreq_log[sa_cnt] <= vp_sa_left_wrreq;
            sa_data_log[sa_cnt]  <= vp_sa_left_data;
            sa_cnt               <= sa_cnt + 1;
        end
    end
end

// Helper: pulse vector_start for one cycle then wait for idle
task vp_start_and_wait;
    input  [2:0] src;
    input  [2:0] dst;
    input [15:0] msrc;
    input [15:0] mdst;
    input [15:0] len;
    input  [3:0] d0;
    input  [3:0] d1;
    integer      t;
    begin
        vp_vect_source        = src;
        vp_vect_dest          = dst;
        vp_mem_source_address = msrc;
        vp_mem_dest_address   = mdst;
        vp_length             = len;
        vp_dim0               = d0;
        vp_dim1               = d1;
        vp_vector_start = 1'b1;
        @(posedge clk); #1;
        vp_vector_start = 1'b0;
        // Wait up to 200 cycles for completion
        for (t = 0; t < 200; t = t + 1) begin
            if (vp_vector_idle) t = 200; // break
            else @(posedge clk); #1;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Main test sequence
// ---------------------------------------------------------------------------
integer j;

initial begin
    // Initialise
    clk              = 0;
    trst             = 1;
    vp_vector_start  = 0;
    vp_vect_source   = 0;
    vp_vect_dest     = 0;
    vp_mem_source_address = 16'd0;
    vp_mem_dest_address   = 16'd0;
    vp_length        = 16'd0;
    vp_dim0          = 4'd0;
    vp_dim1          = 4'd0;
    ev_count         = 0;
    ev_ok            = 1;
    sa_cnt           = 0;
    sa_ok            = 1;
    capture_sa_top   = 0;
    capture_active   = 0;
    neg_mode         = 0;
    vb_head          = 4'd0;
    vb_tail          = 4'd0;
    vb_cnt           = 5'd0;
    vb_q_reg         = 8'd0;
    pass_count       = 0;
    fail_count       = 0;

    for (j = 0; j < 64; j = j + 1) begin wm_mem[j] = 8'd0; buf_mem[j] = 8'd0; end
    for (j = 0; j < 16; j = j + 1) vb_fifo[j] = 8'd0;

    // -----------------------------------------------------------------------
    // TEST 1: Reset — o_vector_idle HIGH immediately after reset
    // -----------------------------------------------------------------------
    do_reset;
    check(vp_vector_idle === 1'b1, 1);

    // -----------------------------------------------------------------------
    // TEST 2: VSRC_MEM → VDST_ACT  (4 elements streamed from WM to Activator)
    //   WM internal 0..3 (ISA 0x0002..0x0005) = 0xA1..0xA4
    //   Expect element_valid HIGH 4 times, o_data = 0xA1, 0xA2, 0xA3, 0xA4
    // -----------------------------------------------------------------------
    wm_mem[0] = 8'hA1; wm_mem[1] = 8'hA2;
    wm_mem[2] = 8'hA3; wm_mem[3] = 8'hA4;
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ACT, 16'h0002, 16'h0002, 16'd4, 4'd0, 4'd0);

    // ev_count and ev_data[] are updated by the always block above
    if (ev_count !== 4)         ev_ok = 0;
    if (ev_data[0] !== 8'hA1)  ev_ok = 0;
    if (ev_data[1] !== 8'hA2)  ev_ok = 0;
    if (ev_data[2] !== 8'hA3)  ev_ok = 0;
    if (ev_data[3] !== 8'hA4)  ev_ok = 0;

    check(ev_ok === 1, 2);

    // -----------------------------------------------------------------------
    // TEST 3: VSRC_MEM → VDST_SA_A  (2x2 matrix rows → top SA buffers)
    //   WM[0..3] = 0x11, 0x12, 0x21, 0x22 (row-major)
    //   Expected sequence: wrreq=0x01/data=0x11, wrreq=0x01/data=0x12,
    //                      wrreq=0x02/data=0x21, wrreq=0x02/data=0x22
    // -----------------------------------------------------------------------
    wm_mem[0] = 8'h11; wm_mem[1] = 8'h12;
    wm_mem[2] = 8'h21; wm_mem[3] = 8'h22;
    sa_cnt = 0; sa_ok = 1;
    capture_sa_top = 1; capture_active = 1;

    vp_start_and_wait(VSRC_MEM, VDST_SA_A, 16'h0002, 16'h0002, 16'd0, 4'd2, 4'd2);
    capture_active = 0;

    if (sa_cnt !== 4)                                     sa_ok = 0;
    if (sa_wrreq_log[0] !== 8'h01 || sa_data_log[0] !== 8'h11) sa_ok = 0;
    if (sa_wrreq_log[1] !== 8'h01 || sa_data_log[1] !== 8'h12) sa_ok = 0;
    if (sa_wrreq_log[2] !== 8'h02 || sa_data_log[2] !== 8'h21) sa_ok = 0;
    if (sa_wrreq_log[3] !== 8'h02 || sa_data_log[3] !== 8'h22) sa_ok = 0;

    check(sa_ok === 1, 3);

    // -----------------------------------------------------------------------
    // TEST 4: VSRC_MEM → VDST_SA_B  (2x2 matrix columns → left SA buffers, stride)
    //   Same WM data. Stride: col0=WM[0,2], col1=WM[1,3]
    //   Expected: wrreq=0x01/0x11, wrreq=0x01/0x21, wrreq=0x02/0x12, wrreq=0x02/0x22
    // -----------------------------------------------------------------------
    sa_cnt = 0; sa_ok = 1;
    capture_sa_top = 0; capture_active = 1;

    vp_start_and_wait(VSRC_MEM, VDST_SA_B, 16'h0002, 16'h0002, 16'd0, 4'd2, 4'd2);
    capture_active = 0;

    if (sa_cnt !== 4)                                     sa_ok = 0;
    if (sa_wrreq_log[0] !== 8'h01 || sa_data_log[0] !== 8'h11) sa_ok = 0;
    if (sa_wrreq_log[1] !== 8'h01 || sa_data_log[1] !== 8'h21) sa_ok = 0;
    if (sa_wrreq_log[2] !== 8'h02 || sa_data_log[2] !== 8'h12) sa_ok = 0;
    if (sa_wrreq_log[3] !== 8'h02 || sa_data_log[3] !== 8'h22) sa_ok = 0;

    check(sa_ok === 1, 4);

    // -----------------------------------------------------------------------
    // TEST 5: VSRC_VEC_BUF → VDST_MEM  (VB read, WM write)
    //   Push 0xBB and 0xCC into VB; VP writes to WM internal addrs 0 & 1
    //   (ISA dest 0x0002 → WM addr 0; ISA dest 0x0003 → WM addr 1)
    // -----------------------------------------------------------------------
    wm_mem[0] = 8'hFF; wm_mem[1] = 8'hFF;  // sentinel: should be overwritten
    vb_push(8'hBB);
    vb_push(8'hCC);

    vp_start_and_wait(VSRC_VEC_BUF, VDST_MEM, 16'd0, 16'h0002, 16'd2, 4'd0, 4'd0);
    @(posedge clk); #1; // let WM write propagate one extra cycle

    check((wm_mem[0] === 8'hBB) && (wm_mem[1] === 8'hCC), 5);

    // -----------------------------------------------------------------------
    // TEST 6: VSRC_SA_OUT → VDST_MEM  (requantize SA outputs, write to WM)
    //   dim0=1, dim1=2 — two outputs:
    //     Element 0: o_sa_row=0, o_sa_col=0 → sa_c=256  → 256>>8=1   → WM[0]=0x01
    //     Element 1: o_sa_row=1, o_sa_col=0 → sa_c=40000 → >127 → sat → WM[1]=0x7F
    //   sa_c_stub driven by the always @(*) block at module level.
    // -----------------------------------------------------------------------
    wm_mem[0] = 8'hFF; wm_mem[1] = 8'hFF;

    vp_start_and_wait(VSRC_SA_OUT, VDST_MEM, 16'd0, 16'h0002, 16'd0, 4'd1, 4'd2);
    @(posedge clk); #1;

    check((wm_mem[0] === 8'h01) && (wm_mem[1] === 8'h7F), 6);

    // -----------------------------------------------------------------------
    // TEST 7: MEM_ERROR — VP stalls on write to ISA address 0x0000
    //   The VP should enter MEM_ERROR (o_vector_idle LOW) and not complete.
    // -----------------------------------------------------------------------
    do_reset;

    vp_vect_source        = VSRC_MEM;
    vp_vect_dest          = VDST_MEM;
    vp_mem_source_address = 16'h0002;
    vp_mem_dest_address   = 16'h0000;  // illegal dest
    vp_length             = 16'd1;
    vp_dim0               = 4'd0;
    vp_dim1               = 4'd0;
    vp_vector_start       = 1'b1;
    @(posedge clk); #1;
    vp_vector_start = 1'b0;

    // Wait enough cycles for MEM_ADDR to have been reached and entered MEM_ERROR
    begin : t7_wait
        integer t;
        for (t = 0; t < 10; t = t + 1) @(posedge clk);
    end
    #1;

    check(vp_vector_idle === 1'b0, 7);

    // -----------------------------------------------------------------------
    // TEST 8: VSRC_MEM (src=0x0001) → VDST_ACT  (0x1 Buffer as source)
    //   buf_mem[0]=0xD1, buf_mem[1]=0xD2 (internal offsets 0 and 1).
    //   ISA source 0x0001: buf_rd_addr = cur_rd_isa - mem_source_lat = elem_cnt.
    //   Expect element_valid twice, o_data = 0xD1 then 0xD2.
    // -----------------------------------------------------------------------
    do_reset;
    buf_mem[0] = 8'hD1; buf_mem[1] = 8'hD2;
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ACT, 16'h0001, 16'h0002, 16'd2, 4'd0, 4'd0);

    if (ev_count !== 2)        ev_ok = 0;
    if (ev_data[0] !== 8'hD1) ev_ok = 0;
    if (ev_data[1] !== 8'hD2) ev_ok = 0;

    check(ev_ok === 1, 8);

    // -----------------------------------------------------------------------
    // TEST 9: VSRC_VEC_BUF → VDST_MEM (dest=0x0001)  (0x1 Buffer as dest)
    //   Push 0xE1, 0xE2 into VB; VP writes to buf_mem internal addrs 0 & 1.
    //   ISA dest 0x0001: buf_wr_addr = cur_wr_isa - mem_dest_lat = elem_cnt.
    // -----------------------------------------------------------------------
    buf_mem[0] = 8'hFF; buf_mem[1] = 8'hFF;  // sentinel: should be overwritten
    vb_push(8'hE1);
    vb_push(8'hE2);

    vp_start_and_wait(VSRC_VEC_BUF, VDST_MEM, 16'd0, 16'h0001, 16'd2, 4'd0, 4'd0);
    @(posedge clk); #1;

    check((buf_mem[0] === 8'hE1) && (buf_mem[1] === 8'hE2), 9);

    // -----------------------------------------------------------------------
    // TEST 10: VSRC_MEM (src=0x0000) → VDST_ACT  (zero source)
    //   ISA source 0x0000 is the hardwired-zero: mem_rd_data is always 0x00
    //   regardless of what is in memory. State machine proceeds normally.
    //   Expect element_valid twice, o_data = 0x00 both times.
    // -----------------------------------------------------------------------
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ACT, 16'h0000, 16'h0002, 16'd2, 4'd0, 4'd0);

    if (ev_count !== 2)        ev_ok = 0;
    if (ev_data[0] !== 8'h00) ev_ok = 0;
    if (ev_data[1] !== 8'h00) ev_ok = 0;

    check(ev_ok === 1, 10);

    // -----------------------------------------------------------------------
    // TEST 11: VSRC_MEM → VDST_ALU_A  (stream WM data to the ALU input A)
    //   WM[0..1] = 0xB1, 0xB2. The VP routes ALU destinations the same way as
    //   the activator: element_valid pulses with o_data = 0xB1, 0xB2.
    // -----------------------------------------------------------------------
    do_reset;
    wm_mem[0] = 8'hB1; wm_mem[1] = 8'hB2;
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ALU_A, 16'h0002, 16'h0002, 16'd2, 4'd0, 4'd0);

    if (ev_count !== 2)        ev_ok = 0;
    if (ev_data[0] !== 8'hB1) ev_ok = 0;
    if (ev_data[1] !== 8'hB2) ev_ok = 0;

    check(ev_ok === 1, 11);

    // -----------------------------------------------------------------------
    // TEST 12: VSRC_SA_OUT → VDST_MEM  (negative requantization saturation)
    //   neg_mode drives MAC(0,0) = -40000; -40000 >>> 8 = -157 < -128, so the
    //   requantizer saturates to -128 = 0x80. dim0=1, dim1=1 (single element).
    // -----------------------------------------------------------------------
    do_reset;
    neg_mode = 1'b1;
    wm_mem[0] = 8'hFF;  // sentinel

    vp_start_and_wait(VSRC_SA_OUT, VDST_MEM, 16'd0, 16'h0002, 16'd0, 4'd1, 4'd1);
    @(posedge clk); #1;

    check(wm_mem[0] === 8'h80, 12);
    neg_mode = 1'b0;

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    $display("----------------------------------------");
    $display("Results: %0d / %0d tests passed",
             pass_count, pass_count + fail_count);
    $display("----------------------------------------");
    $stop;
end

endmodule
