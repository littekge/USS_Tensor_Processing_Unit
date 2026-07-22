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
 *   All 23 tests should show PASS for a correct implementation.
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
 *   Test 13: VSRC_SA_OUT → VDST_MEM — round-to-nearest bias (M0=1,n=2: 62→16)
 *   Test 14: VSRC_SA_OUT → VDST_MEM — M0 scale multiply (M0=25,n=3: 10→31)
 *   Test 15 (v0.5): VSRC_MEM → VDST_SA_A with leading_dimension=4 — reads a 2x2
 *            top-left sub-block of a 4-wide parent in place (rows spaced by the
 *            leading dimension, not the sub-block width).
 *   Test 16 (v0.5): VSRC_SA_OUT → VDST_MEM with leading_dimension=4 — writes a
 *            2x2 result into a 4-wide parent in place (row 1 lands at base+4,
 *            not base+2); leading_dimension=0 (Tests 3/4) is the contiguous
 *            v0.4 behavior.
 *   Test 17 (v0.6): im2col gather with zero-fill on a small padded map; dense
 *            matrix written Row-major to dest.
 *   Test 18 (v0.6): max window stream to the Pooler destination with window_end
 *            on each window's final element; descriptor-derived count.
 *   Test 19 (v0.6): max window stream with min-fill (0x80) on padding.
 *   Test 20 (v0.6 fix): im2col gather sourced from the 0x1 staging buffer
 *            (rs1=0x1). A 3x3 feature map staged at buffer offsets 0..8 is
 *            expanded (2x2 window, stride 1). The windowed path must hold the
 *            flat address at 0x1 and walk the buffer via the offset (exactly as
 *            the non-windowed 0x1 path in Tests 8/9); the pre-fix RTL baked the
 *            index into the address, so only offset 0 read the buffer and every
 *            other element wrongly decoded to main memory.
 *   Test 21 (v0.6 fix): max window stream sourced from the 0x1 buffer (rs1=0x1);
 *            same 0x1-buffer walk on the pooling path.
 *   Test 22 (v0.7): move — contiguous device-memory copy main->main of `len`
 *            elements (VSRC_MEM -> VDST_MEM), verified element-by-element; the
 *            source region is left intact.
 *   Test 23 (v0.7): move output staging main->0x1 — copies a result tensor from
 *            main memory into the isolated 0x1 I/O buffer (dest walks the buffer
 *            offset), and confirms main memory outside the source (a weight
 *            sentinel) is untouched by the staged write.
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
reg [23:0] vp_mem_source_address;
reg [23:0] vp_mem_dest_address;
reg [15:0] vp_length;
reg  [3:0] vp_dim0;
reg  [3:0] vp_dim1;
reg  [7:0] vp_scale;   // M0: dyadic requant multiplier (VSRC_SA_OUT tests)
reg  [7:0] vp_shift;   // n:  dyadic requant right-shift
reg [23:0] vp_leading_dimension;  // physical row stride (0 = contiguous)

// v0.6 window descriptor drive registers (windowed tests only)
reg [11:0] vp_win_chans, vp_win_inh, vp_win_inw, vp_win_outh, vp_win_outw;
reg [3:0]  vp_win_winh, vp_win_winw, vp_win_strh, vp_win_strw, vp_win_padh, vp_win_padw;

wire        vp_vector_idle;
wire        vp_element_valid;
wire        vp_window_end;
// v0.4: single unified Data_Memory port (flat 24-bit address + 0x1 offset).
wire [23:0] vp_dm_address;
wire        vp_dm_wren;
wire  [7:0] vp_dm_data;
wire  [9:0] vp_dm_offset;
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
localparam VDST_POOL    = 3'd6;
localparam VSRC_MEM_WIN = 3'd3;

// ---------------------------------------------------------------------------
// Behavioral Data_Memory model (v0.4 unified wrapper)
//
// Mirrors the real Data_Memory decode as seen by the Vector_Processor:
//   flat address 0x0        -> hardwired zero (read returns 0)
//   flat address 0x1        -> 0x1 buffer, indexed by the offset port
//   any other flat address  -> main data memory, indexed by the flat address
// The wrapper has a 2-cycle registered read latency (registered address +
// registered output), which is exactly why the VP has two memory wait states;
// this model reproduces it with a two-stage output pipeline so the timing is
// faithful. main_mem is indexed by the low bits of the flat address (the test
// programs use small addresses), and buf_mem by the 0x1 offset port.
// ---------------------------------------------------------------------------
reg [7:0] main_mem [0:1023];  // flat main data memory
reg [7:0] buf_mem  [0:1023];  // 0x1 I/O-staging buffer

wire dm_is_zero = (vp_dm_address == 24'd0);
wire dm_is_buf  = (vp_dm_address == 24'd1);

// Combinational read selection (pre-latency), matching the wrapper decode.
wire [7:0] dm_read_sel = dm_is_zero ? 8'd0 :
                         dm_is_buf  ? buf_mem[vp_dm_offset] :
                         main_mem[vp_dm_address[9:0]];

reg [7:0] dm_q_stage1, dm_q_reg;
wire [7:0] dm_q = dm_q_reg;

always @(posedge clk) begin
    if (vp_dm_wren) begin
        if (dm_is_buf)        buf_mem[vp_dm_offset]        <= vp_dm_data;
        else if (!dm_is_zero) main_mem[vp_dm_address[9:0]] <= vp_dm_data;
    end
    // Two-stage output pipeline -> 2-cycle registered read latency.
    dm_q_stage1 <= dm_read_sel;
    dm_q_reg    <= dm_q_stage1;
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

// force_mode (Tests 13/14): drives MAC(0,0) to an arbitrary value so the
// dyadic requant datapath (scale multiply + rounding bias) can be checked
// against hand-computed results.
reg               force_mode;
reg signed [31:0] sa_c_force;

// stride_mode (Test 16): drives each MAC to a distinct small value so strided
// writeback placement can be verified. The VP reads MAC(sa_row=col,sa_col=row)
// for output[row][col], so value = 10 + 2*sa_col + sa_row maps output[row][col]
// to 10,11,12,13 for (0,0),(0,1),(1,0),(1,1) respectively.
reg stride_mode;

// Selects specific values used in Test 6:
//   MAC(row=0,col=0) → 256  → requantized to 0x01
//   MAC(row=1,col=0) → 40000 → saturates to  0x7F
// and in Test 12 (neg_mode): MAC(0,0) → -40000 → saturates to 0x80.
always @(*) begin
    if      (stride_mode) sa_c_stub = 32'd10 + (2 * {28'd0, vp_sa_col}) + {28'd0, vp_sa_row};
    else if (force_mode && vp_sa_row == 4'd0 && vp_sa_col == 4'd0) sa_c_stub = sa_c_force;
    else if (neg_mode && vp_sa_row == 4'd0 && vp_sa_col == 4'd0) sa_c_stub = -32'sd40000;
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
    .i_leading_dimension  (vp_leading_dimension),
    .i_scale              (vp_scale),
    .i_shift              (vp_shift),
    .i_win_chans          (vp_win_chans),
    .i_win_inh            (vp_win_inh),
    .i_win_inw            (vp_win_inw),
    .i_win_outh           (vp_win_outh),
    .i_win_outw           (vp_win_outw),
    .i_win_winh           (vp_win_winh),
    .i_win_winw           (vp_win_winw),
    .i_win_strh           (vp_win_strh),
    .i_win_strw           (vp_win_strw),
    .i_win_padh           (vp_win_padh),
    .i_win_padw           (vp_win_padw),
    .o_vector_idle        (vp_vector_idle),
    .o_element_valid      (vp_element_valid),
    .o_window_end         (vp_window_end),
    .o_dm_address         (vp_dm_address),
    .o_dm_wren            (vp_dm_wren),
    .o_dm_data            (vp_dm_data),
    .o_dm_offset          (vp_dm_offset),
    .i_dm_q               (dm_q),
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

// v0.3: requantization parameters M0/n are driven at runtime via vp_scale /
// vp_shift (per the MUL instruction) rather than a compile-time parameter.
// The SA-output tests set them explicitly before each run.

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

// Windowed (max) stream capture: logs the element stream to the Pooler and
// whether o_window_end coincided with each element_valid pulse.
integer   win_ev_count;
reg [7:0] win_stream [0:31];
reg       win_wend   [0:31];
reg       capture_win;

always @(posedge clk) begin
    if (capture_win && vp_element_valid && win_ev_count < 32) begin
        win_stream[win_ev_count] <= vp_data;
        win_wend[win_ev_count]   <= vp_window_end;
        win_ev_count             <= win_ev_count + 1;
    end
end

// Helper: pulse vector_start for one cycle then wait for idle
task vp_start_and_wait;
    input  [2:0] src;
    input  [2:0] dst;
    input [23:0] msrc;
    input [23:0] mdst;
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
    vp_mem_source_address = 24'd0;
    vp_mem_dest_address   = 24'd0;
    vp_length        = 16'd0;
    vp_dim0          = 4'd0;
    vp_dim1          = 4'd0;
    vp_scale         = 8'd1;   // identity requant multiplier by default
    vp_shift         = 8'd0;   // no shift by default (non-SA-out tests ignore)
    vp_leading_dimension = 24'd0;  // contiguous by default (v0.4 behavior)
    vp_win_chans = 12'd0; vp_win_inh = 12'd0; vp_win_inw = 12'd0;
    vp_win_outh = 12'd0; vp_win_outw = 12'd0;
    vp_win_winh = 4'd0; vp_win_winw = 4'd0; vp_win_strh = 4'd0;
    vp_win_strw = 4'd0; vp_win_padh = 4'd0; vp_win_padw = 4'd0;
    capture_win      = 0;
    win_ev_count     = 0;
    stride_mode      = 0;
    ev_count         = 0;
    ev_ok            = 1;
    sa_cnt           = 0;
    sa_ok            = 1;
    capture_sa_top   = 0;
    capture_active   = 0;
    neg_mode         = 0;
    force_mode       = 0;
    sa_c_force       = 32'd0;
    vb_head          = 4'd0;
    vb_tail          = 4'd0;
    vb_cnt           = 5'd0;
    vb_q_reg         = 8'd0;
    pass_count       = 0;
    fail_count       = 0;

    for (j = 0; j < 1024; j = j + 1) begin main_mem[j] = 8'd0; buf_mem[j] = 8'd0; end
    for (j = 0; j < 16; j = j + 1) vb_fifo[j] = 8'd0;

    // -----------------------------------------------------------------------
    // TEST 1: Reset — o_vector_idle HIGH immediately after reset
    // -----------------------------------------------------------------------
    do_reset;
    check(vp_vector_idle === 1'b1, 1);

    // -----------------------------------------------------------------------
    // TEST 2: VSRC_MEM → VDST_ACT  (4 elements streamed from memory to Activator)
    //   Flat main-memory 0x2..0x5 = 0xA1..0xA4
    //   Expect element_valid HIGH 4 times, o_data = 0xA1, 0xA2, 0xA3, 0xA4
    // -----------------------------------------------------------------------
    main_mem[2] = 8'hA1; main_mem[3] = 8'hA2;
    main_mem[4] = 8'hA3; main_mem[5] = 8'hA4;
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ACT, 24'h000002, 24'h000002, 16'd4, 4'd0, 4'd0);

    // ev_count and ev_data[] are updated by the always block above
    if (ev_count !== 4)         ev_ok = 0;
    if (ev_data[0] !== 8'hA1)  ev_ok = 0;
    if (ev_data[1] !== 8'hA2)  ev_ok = 0;
    if (ev_data[2] !== 8'hA3)  ev_ok = 0;
    if (ev_data[3] !== 8'hA4)  ev_ok = 0;

    check(ev_ok === 1, 2);

    // -----------------------------------------------------------------------
    // TEST 3: VSRC_MEM → VDST_SA_A  (2x2 matrix rows → top SA buffers)
    //   Flat main-memory 0x2..0x5 = 0x11, 0x12, 0x21, 0x22 (row-major)
    //   Expected sequence: wrreq=0x01/data=0x11, wrreq=0x01/data=0x12,
    //                      wrreq=0x02/data=0x21, wrreq=0x02/data=0x22
    // -----------------------------------------------------------------------
    main_mem[2] = 8'h11; main_mem[3] = 8'h12;
    main_mem[4] = 8'h21; main_mem[5] = 8'h22;
    sa_cnt = 0; sa_ok = 1;
    capture_sa_top = 1; capture_active = 1;

    vp_start_and_wait(VSRC_MEM, VDST_SA_A, 24'h000002, 24'h000002, 16'd0, 4'd2, 4'd2);
    capture_active = 0;

    if (sa_cnt !== 4)                                     sa_ok = 0;
    if (sa_wrreq_log[0] !== 8'h01 || sa_data_log[0] !== 8'h11) sa_ok = 0;
    if (sa_wrreq_log[1] !== 8'h01 || sa_data_log[1] !== 8'h12) sa_ok = 0;
    if (sa_wrreq_log[2] !== 8'h02 || sa_data_log[2] !== 8'h21) sa_ok = 0;
    if (sa_wrreq_log[3] !== 8'h02 || sa_data_log[3] !== 8'h22) sa_ok = 0;

    check(sa_ok === 1, 3);

    // -----------------------------------------------------------------------
    // TEST 4: VSRC_MEM → VDST_SA_B  (2x2 matrix columns → left SA buffers, stride)
    //   Same main-memory data. Stride: col0=mem[0,2], col1=mem[1,3]
    //   Expected: wrreq=0x01/0x11, wrreq=0x01/0x21, wrreq=0x02/0x12, wrreq=0x02/0x22
    // -----------------------------------------------------------------------
    sa_cnt = 0; sa_ok = 1;
    capture_sa_top = 0; capture_active = 1;

    vp_start_and_wait(VSRC_MEM, VDST_SA_B, 24'h000002, 24'h000002, 16'd0, 4'd2, 4'd2);
    capture_active = 0;

    if (sa_cnt !== 4)                                     sa_ok = 0;
    if (sa_wrreq_log[0] !== 8'h01 || sa_data_log[0] !== 8'h11) sa_ok = 0;
    if (sa_wrreq_log[1] !== 8'h01 || sa_data_log[1] !== 8'h21) sa_ok = 0;
    if (sa_wrreq_log[2] !== 8'h02 || sa_data_log[2] !== 8'h12) sa_ok = 0;
    if (sa_wrreq_log[3] !== 8'h02 || sa_data_log[3] !== 8'h22) sa_ok = 0;

    check(sa_ok === 1, 4);

    // -----------------------------------------------------------------------
    // TEST 5: VSRC_VEC_BUF → VDST_MEM  (VB read, main-memory write)
    //   Push 0xBB and 0xCC into VB; VP writes to flat main-memory 0x2 & 0x3.
    // -----------------------------------------------------------------------
    main_mem[2] = 8'hFF; main_mem[3] = 8'hFF;  // sentinel: should be overwritten
    vb_push(8'hBB);
    vb_push(8'hCC);

    vp_start_and_wait(VSRC_VEC_BUF, VDST_MEM, 24'd0, 24'h000002, 16'd2, 4'd0, 4'd0);
    @(posedge clk); #1; // let the memory write propagate one extra cycle

    check((main_mem[2] === 8'hBB) && (main_mem[3] === 8'hCC), 5);

    // -----------------------------------------------------------------------
    // TEST 6: VSRC_SA_OUT → VDST_MEM  (requantize SA outputs, write to memory)
    //   dim0=1, dim1=2 — two outputs written to flat main-memory 0x2 & 0x3:
    //     Element 0: o_sa_row=0, o_sa_col=0 → sa_c=256, M0=1, n=8
    //                (256 + (1<<7))>>8 = 384>>8 = 1        → mem[2]=0x01
    //     Element 1: o_sa_row=1, o_sa_col=0 → sa_c=40000
    //                (40000 + 128)>>8 = 156 → >127 → sat   → mem[3]=0x7F
    //   sa_c_stub driven by the always @(*) block at module level.
    // -----------------------------------------------------------------------
    main_mem[2] = 8'hFF; main_mem[3] = 8'hFF;
    vp_scale = 8'd1; vp_shift = 8'd8;

    vp_start_and_wait(VSRC_SA_OUT, VDST_MEM, 24'd0, 24'h000002, 16'd0, 4'd1, 4'd2);
    @(posedge clk); #1;

    check((main_mem[2] === 8'h01) && (main_mem[3] === 8'h7F), 6);

    // -----------------------------------------------------------------------
    // TEST 7: MEM_ERROR — VP stalls on write to reserved address 0x0
    //   The VP should enter MEM_ERROR (o_vector_idle LOW) and not complete.
    // -----------------------------------------------------------------------
    do_reset;

    vp_vect_source        = VSRC_MEM;
    vp_vect_dest          = VDST_MEM;
    vp_mem_source_address = 24'h000002;
    vp_mem_dest_address   = 24'h000000;  // illegal dest
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
    //   buf_mem[0]=0xD1, buf_mem[1]=0xD2 (buffer offsets 0 and 1).
    //   ISA source 0x0001: flat address held at 0x1, o_dm_offset walks 0,1.
    //   Expect element_valid twice, o_data = 0xD1 then 0xD2.
    // -----------------------------------------------------------------------
    do_reset;
    buf_mem[0] = 8'hD1; buf_mem[1] = 8'hD2;
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ACT, 24'h000001, 24'h000002, 16'd2, 4'd0, 4'd0);

    if (ev_count !== 2)        ev_ok = 0;
    if (ev_data[0] !== 8'hD1) ev_ok = 0;
    if (ev_data[1] !== 8'hD2) ev_ok = 0;

    check(ev_ok === 1, 8);

    // -----------------------------------------------------------------------
    // TEST 9: VSRC_VEC_BUF → VDST_MEM (dest=0x0001)  (0x1 Buffer as dest)
    //   Push 0xE1, 0xE2 into VB; VP writes to buffer offsets 0 & 1.
    //   ISA dest 0x0001: flat address held at 0x1, o_dm_offset walks 0,1.
    // -----------------------------------------------------------------------
    buf_mem[0] = 8'hFF; buf_mem[1] = 8'hFF;  // sentinel: should be overwritten
    vb_push(8'hE1);
    vb_push(8'hE2);

    vp_start_and_wait(VSRC_VEC_BUF, VDST_MEM, 24'd0, 24'h000001, 16'd2, 4'd0, 4'd0);
    @(posedge clk); #1;

    check((buf_mem[0] === 8'hE1) && (buf_mem[1] === 8'hE2), 9);

    // -----------------------------------------------------------------------
    // TEST 10: VSRC_MEM (src=0x0000) → VDST_ACT  (zero source)
    //   ISA source 0x0000 is the hardwired-zero: mem_rd_data is always 0x00
    //   regardless of what is in memory. State machine proceeds normally.
    //   Expect element_valid twice, o_data = 0x00 both times.
    // -----------------------------------------------------------------------
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ACT, 24'h000000, 24'h000002, 16'd2, 4'd0, 4'd0);

    if (ev_count !== 2)        ev_ok = 0;
    if (ev_data[0] !== 8'h00) ev_ok = 0;
    if (ev_data[1] !== 8'h00) ev_ok = 0;

    check(ev_ok === 1, 10);

    // -----------------------------------------------------------------------
    // TEST 11: VSRC_MEM → VDST_ALU_A  (stream memory data to the ALU input A)
    //   Flat main-memory 0x2..0x3 = 0xB1, 0xB2. The VP routes ALU destinations
    //   the same way as the activator: element_valid pulses o_data = 0xB1, 0xB2.
    // -----------------------------------------------------------------------
    do_reset;
    main_mem[2] = 8'hB1; main_mem[3] = 8'hB2;
    ev_count = 0; ev_ok = 1;

    vp_start_and_wait(VSRC_MEM, VDST_ALU_A, 24'h000002, 24'h000002, 16'd2, 4'd0, 4'd0);

    if (ev_count !== 2)        ev_ok = 0;
    if (ev_data[0] !== 8'hB1) ev_ok = 0;
    if (ev_data[1] !== 8'hB2) ev_ok = 0;

    check(ev_ok === 1, 11);

    // -----------------------------------------------------------------------
    // TEST 12: VSRC_SA_OUT → VDST_MEM  (negative requantization saturation)
    //   neg_mode drives MAC(0,0) = -40000; with M0=1, n=8 the requant is
    //   (-40000 + 128) >>> 8 = -156 < -128, so it saturates to -128 = 0x80.
    //   dim0=1, dim1=1 (single element).
    // -----------------------------------------------------------------------
    do_reset;
    neg_mode = 1'b1;
    vp_scale = 8'd1; vp_shift = 8'd8;
    main_mem[2] = 8'hFF;  // sentinel

    vp_start_and_wait(VSRC_SA_OUT, VDST_MEM, 24'd0, 24'h000002, 16'd0, 4'd1, 4'd1);
    @(posedge clk); #1;

    check(main_mem[2] === 8'h80, 12);
    neg_mode = 1'b0;

    // -----------------------------------------------------------------------
    // TEST 13: VSRC_SA_OUT → VDST_MEM  (round-to-nearest via the rounding bias)
    //   force_mode drives MAC(0,0) = 62; M0=1, n=2.
    //     result = (1*62 + (1<<1)) >> 2 = (62 + 2) >> 2 = 64 >> 2 = 16 = 0x10.
    //   A truncating shift (no bias) would give 62>>2 = 15 = 0x0F, so this test
    //   distinguishes round-to-nearest from truncation.
    // -----------------------------------------------------------------------
    do_reset;
    force_mode = 1'b1;
    sa_c_force = 32'sd62;
    vp_scale = 8'd1; vp_shift = 8'd2;
    main_mem[2] = 8'hFF;  // sentinel

    vp_start_and_wait(VSRC_SA_OUT, VDST_MEM, 24'd0, 24'h000002, 16'd0, 4'd1, 4'd1);
    @(posedge clk); #1;

    check(main_mem[2] === 8'h10, 13);

    // -----------------------------------------------------------------------
    // TEST 14: VSRC_SA_OUT → VDST_MEM  (M0 scale multiply)
    //   force_mode drives MAC(0,0) = 10; M0=25, n=3.
    //     result = (25*10 + (1<<2)) >> 3 = (250 + 4) >> 3 = 254 >> 3 = 31 = 0x1F.
    //   Verifies the unsigned scale multiply feeds requantization (a bare shift
    //   of 10 could never reach 31).
    // -----------------------------------------------------------------------
    do_reset;
    force_mode = 1'b1;
    sa_c_force = 32'sd10;
    vp_scale = 8'd25; vp_shift = 8'd3;
    main_mem[2] = 8'hFF;  // sentinel

    vp_start_and_wait(VSRC_SA_OUT, VDST_MEM, 24'd0, 24'h000002, 16'd0, 4'd1, 4'd1);
    @(posedge clk); #1;

    check(main_mem[2] === 8'h1F, 14);
    force_mode = 1'b0;

    // -----------------------------------------------------------------------
    // TEST 15 (v0.5): VSRC_MEM → VDST_SA_A with leading_dimension=4.
    //   A 4x4 parent matrix is stored contiguously from flat address 0x2. A 2x2
    //   top-left sub-block is read in place, so row 1 begins at base+4 (not
    //   base+2 as contiguous width-2 addressing would give).
    //     parent row 0: mem[2..5] = 0x11,0x12,0x13,0x14
    //     parent row 1: mem[6..9] = 0x21,0x22,0x23,0x24
    //   Expected push sequence (top buffers): 0x11,0x12 (row0), 0x21,0x22 (row1).
    // -----------------------------------------------------------------------
    do_reset;
    main_mem[2] = 8'h11; main_mem[3] = 8'h12; main_mem[4] = 8'h13; main_mem[5] = 8'h14;
    main_mem[6] = 8'h21; main_mem[7] = 8'h22; main_mem[8] = 8'h23; main_mem[9] = 8'h24;
    sa_cnt = 0; sa_ok = 1;
    capture_sa_top = 1; capture_active = 1;
    vp_leading_dimension = 24'd4;

    vp_start_and_wait(VSRC_MEM, VDST_SA_A, 24'h000002, 24'h000002, 16'd0, 4'd2, 4'd2);
    capture_active = 0;
    vp_leading_dimension = 24'd0;

    if (sa_cnt !== 4)                                          sa_ok = 0;
    if (sa_wrreq_log[0] !== 8'h01 || sa_data_log[0] !== 8'h11) sa_ok = 0;
    if (sa_wrreq_log[1] !== 8'h01 || sa_data_log[1] !== 8'h12) sa_ok = 0;
    if (sa_wrreq_log[2] !== 8'h02 || sa_data_log[2] !== 8'h21) sa_ok = 0;
    if (sa_wrreq_log[3] !== 8'h02 || sa_data_log[3] !== 8'h22) sa_ok = 0;

    check(sa_ok === 1, 15);

    // -----------------------------------------------------------------------
    // TEST 16 (v0.5): VSRC_SA_OUT → VDST_MEM with leading_dimension=4.
    //   A 2x2 result is written into a 4-wide parent starting at flat 0x2, so
    //   result row 1 lands at base+4 (mem[6..7]) not base+2 (mem[4..5]). With
    //   M0=1,n=0 the requant is the identity, and stride_mode drives distinct
    //   values so placement is unambiguous:
    //     out[0][0]=10→mem[2]  out[0][1]=11→mem[3]
    //     out[1][0]=12→mem[6]  out[1][1]=13→mem[7]
    //   mem[4],mem[5] (parent row-0 cols 2,3) stay at the sentinel 0xFF.
    // -----------------------------------------------------------------------
    do_reset;
    stride_mode = 1'b1;
    vp_scale = 8'd1; vp_shift = 8'd0;
    vp_leading_dimension = 24'd4;
    main_mem[2] = 8'hFF; main_mem[3] = 8'hFF; main_mem[4] = 8'hFF; main_mem[5] = 8'hFF;
    main_mem[6] = 8'hFF; main_mem[7] = 8'hFF;

    vp_start_and_wait(VSRC_SA_OUT, VDST_MEM, 24'd0, 24'h000002, 16'd0, 4'd2, 4'd2);
    @(posedge clk); #1;

    check((main_mem[2] === 8'd10) && (main_mem[3] === 8'd11) &&
          (main_mem[6] === 8'd12) && (main_mem[7] === 8'd13) &&
          (main_mem[4] === 8'hFF) && (main_mem[5] === 8'hFF), 16);
    stride_mode = 1'b0;
    vp_leading_dimension = 24'd0;

    // -----------------------------------------------------------------------
    // TEST 17 (v0.6): im2col gather with zero-fill (padded).
    //   Feature map chans=1, inh=2, inw=2 = [[1,2],[3,4]] at src 0x10; window
    //   2x2, stride 2, pad 1 -> outh=outw=2. The dense (winh*winw)=4 x
    //   (outh*outw)=4 im2col matrix is written Row-major to dest 0x40, with
    //   out-of-bounds (padding) elements zero-filled. Hand-computed dense
    //   (r = window offset, c = output position; element order r-major, c-inner):
    //     [0,0,0,4, 0,0,3,0, 0,2,0,0, 1,0,0,0]
    // -----------------------------------------------------------------------
    do_reset;
    main_mem[16] = 8'd1; main_mem[17] = 8'd2;
    main_mem[18] = 8'd3; main_mem[19] = 8'd4;
    for (j = 64; j < 80; j = j + 1) main_mem[j] = 8'hFF;
    vp_win_chans = 12'd1; vp_win_inh = 12'd2; vp_win_inw = 12'd2;
    vp_win_outh  = 12'd2; vp_win_outw = 12'd2;
    vp_win_winh = 4'd2; vp_win_winw = 4'd2; vp_win_strh = 4'd2;
    vp_win_strw = 4'd2; vp_win_padh = 4'd1; vp_win_padw = 4'd1;

    vp_start_and_wait(VSRC_MEM_WIN, VDST_MEM, 24'h000010, 24'h000040,
                      16'd0, 4'd0, 4'd0);
    @(posedge clk); #1;

    begin : im2col_check
        reg       ok;
        integer   k;
        reg [7:0] exp [0:15];
        exp[0]=8'd0;  exp[1]=8'd0;  exp[2]=8'd0;  exp[3]=8'd4;
        exp[4]=8'd0;  exp[5]=8'd0;  exp[6]=8'd3;  exp[7]=8'd0;
        exp[8]=8'd0;  exp[9]=8'd2;  exp[10]=8'd0; exp[11]=8'd0;
        exp[12]=8'd1; exp[13]=8'd0; exp[14]=8'd0; exp[15]=8'd0;
        ok = 1'b1;
        for (k = 0; k < 16; k = k + 1)
            if (main_mem[64 + k] !== exp[k]) ok = 1'b0;
        check(ok === 1'b1, 17);
    end

    // -----------------------------------------------------------------------
    // TEST 18 (v0.6): max window stream to the Pooler destination + window_end.
    //   Feature map chans=1, inh=4, inw=4 (values 0..15 Row-major) at src 0x20;
    //   window 2x2, stride 2, pad 0 -> outh=outw=2, four windows of four values.
    //   Streamed element order (window inner, wc innermost; windows in ch,oh,ow
    //   order); window_end asserted on each window's 4th element:
    //     0,1,4,5 | 2,3,6,7 | 8,9,12,13 | 10,11,14,15
    // -----------------------------------------------------------------------
    do_reset;
    for (j = 0; j < 16; j = j + 1) main_mem[32 + j] = j;
    vp_win_chans = 12'd1; vp_win_inh = 12'd4; vp_win_inw = 12'd4;
    vp_win_outh  = 12'd2; vp_win_outw = 12'd2;
    vp_win_winh = 4'd2; vp_win_winw = 4'd2; vp_win_strh = 4'd2;
    vp_win_strw = 4'd2; vp_win_padh = 4'd0; vp_win_padw = 4'd0;
    capture_win = 1'b1; win_ev_count = 0;

    vp_start_and_wait(VSRC_MEM_WIN, VDST_POOL, 24'h000020, 24'd0,
                      16'd0, 4'd0, 4'd0);
    capture_win = 1'b0;

    begin : max_check
        reg       ok;
        integer   k;
        reg [7:0] exp [0:15];
        exp[0]=8'd0;  exp[1]=8'd1;  exp[2]=8'd4;  exp[3]=8'd5;
        exp[4]=8'd2;  exp[5]=8'd3;  exp[6]=8'd6;  exp[7]=8'd7;
        exp[8]=8'd8;  exp[9]=8'd9;  exp[10]=8'd12; exp[11]=8'd13;
        exp[12]=8'd10; exp[13]=8'd11; exp[14]=8'd14; exp[15]=8'd15;
        ok = 1'b1;
        if (win_ev_count !== 16) ok = 1'b0;
        for (k = 0; k < 16; k = k + 1)
        begin
            if (win_stream[k] !== exp[k]) ok = 1'b0;
            if (((k % 4) == 3) && (win_wend[k] !== 1'b1)) ok = 1'b0;
            if (((k % 4) != 3) && (win_wend[k] !== 1'b0)) ok = 1'b0;
        end
        check(ok === 1'b1, 18);
    end

    // -----------------------------------------------------------------------
    // TEST 19 (v0.6): max window stream with min-fill on padding.
    //   Feature map chans=1, inh=2, inw=2 = [[1,2],[3,4]] at src 0x10; window
    //   2x2, stride 2, pad 1 (same geometry as Test 17). Out-of-bounds elements
    //   take the min representable value (0x80) so they never win the max:
    //     80,80,80,1 | 80,80,2,80 | 80,3,80,80 | 4,80,80,80
    //   window_end on each window's 4th element.
    // -----------------------------------------------------------------------
    do_reset;
    main_mem[16] = 8'd1; main_mem[17] = 8'd2;
    main_mem[18] = 8'd3; main_mem[19] = 8'd4;
    vp_win_chans = 12'd1; vp_win_inh = 12'd2; vp_win_inw = 12'd2;
    vp_win_outh  = 12'd2; vp_win_outw = 12'd2;
    vp_win_winh = 4'd2; vp_win_winw = 4'd2; vp_win_strh = 4'd2;
    vp_win_strw = 4'd2; vp_win_padh = 4'd1; vp_win_padw = 4'd1;
    capture_win = 1'b1; win_ev_count = 0;

    vp_start_and_wait(VSRC_MEM_WIN, VDST_POOL, 24'h000010, 24'd0,
                      16'd0, 4'd0, 4'd0);
    capture_win = 1'b0;

    begin : maxfill_check
        reg       ok;
        integer   k;
        reg [7:0] exp [0:15];
        exp[0]=8'h80; exp[1]=8'h80; exp[2]=8'h80; exp[3]=8'd1;
        exp[4]=8'h80; exp[5]=8'h80; exp[6]=8'd2;  exp[7]=8'h80;
        exp[8]=8'h80; exp[9]=8'd3;  exp[10]=8'h80; exp[11]=8'h80;
        exp[12]=8'd4; exp[13]=8'h80; exp[14]=8'h80; exp[15]=8'h80;
        ok = 1'b1;
        if (win_ev_count !== 16) ok = 1'b0;
        for (k = 0; k < 16; k = k + 1)
        begin
            if (win_stream[k] !== exp[k]) ok = 1'b0;
            if (((k % 4) == 3) && (win_wend[k] !== 1'b1)) ok = 1'b0;
            if (((k % 4) != 3) && (win_wend[k] !== 1'b0)) ok = 1'b0;
        end
        check(ok === 1'b1, 19);
    end

    // -----------------------------------------------------------------------
    // TEST 20 (v0.6 fix): im2col gather sourced from the 0x1 staging buffer.
    //   A 1-channel 3x3 feature map is staged in the 0x1 buffer at offsets 0..8
    //   ([[1,2,3],[4,5,6],[7,8,9]]); window 2x2, stride 1, pad 0 -> outh=outw=2.
    //   The dense (winh*winw)=4 x (outh*outw)=4 im2col matrix is written
    //   Row-major to main-memory dest 0x40. Because the source is 0x1, the
    //   windowed read must hold the address at 0x1 and index the buffer via the
    //   offset; a flat-address walk (pre-fix) would read main memory for every
    //   element past the first. Hand-computed dense (row k = wr*winw+wc,
    //   column p = oy*outw+ox; k-major, p-inner):
    //     [1,2,4,5, 2,3,5,6, 4,5,7,8, 5,6,8,9]
    // -----------------------------------------------------------------------
    do_reset;
    for (j = 0; j < 9; j = j + 1) buf_mem[j] = j + 1;   // 1..9 at offsets 0..8
    for (j = 64; j < 80; j = j + 1) main_mem[j] = 8'hFF;
    vp_win_chans = 12'd1; vp_win_inh = 12'd3; vp_win_inw = 12'd3;
    vp_win_outh  = 12'd2; vp_win_outw = 12'd2;
    vp_win_winh = 4'd2; vp_win_winw = 4'd2; vp_win_strh = 4'd1;
    vp_win_strw = 4'd1; vp_win_padh = 4'd0; vp_win_padw = 4'd0;

    vp_start_and_wait(VSRC_MEM_WIN, VDST_MEM, 24'h000001, 24'h000040,
                      16'd0, 4'd0, 4'd0);
    @(posedge clk); #1;

    begin : im2col_buf_check
        reg       ok;
        integer   k;
        reg [7:0] exp [0:15];
        exp[0]=8'd1;  exp[1]=8'd2;  exp[2]=8'd4;  exp[3]=8'd5;
        exp[4]=8'd2;  exp[5]=8'd3;  exp[6]=8'd5;  exp[7]=8'd6;
        exp[8]=8'd4;  exp[9]=8'd5;  exp[10]=8'd7; exp[11]=8'd8;
        exp[12]=8'd5; exp[13]=8'd6; exp[14]=8'd8; exp[15]=8'd9;
        ok = 1'b1;
        for (k = 0; k < 16; k = k + 1)
            if (main_mem[64 + k] !== exp[k]) ok = 1'b0;
        check(ok === 1'b1, 20);
    end

    // -----------------------------------------------------------------------
    // TEST 21 (v0.6 fix): max window stream sourced from the 0x1 buffer.
    //   Same 3x3 feature map staged in the 0x1 buffer at offsets 0..8; window
    //   2x2, stride 1, pad 0 -> four windows of four values streamed to the
    //   Pooler destination, window_end on each window's 4th element. Streamed
    //   order (window inner wc innermost; windows in ch,oh,ow order):
    //     1,2,4,5 | 2,3,5,6 | 4,5,7,8 | 5,6,8,9
    //   Verifies the 0x1-buffer walk on the pooling (max) path.
    // -----------------------------------------------------------------------
    do_reset;
    for (j = 0; j < 9; j = j + 1) buf_mem[j] = j + 1;
    vp_win_chans = 12'd1; vp_win_inh = 12'd3; vp_win_inw = 12'd3;
    vp_win_outh  = 12'd2; vp_win_outw = 12'd2;
    vp_win_winh = 4'd2; vp_win_winw = 4'd2; vp_win_strh = 4'd1;
    vp_win_strw = 4'd1; vp_win_padh = 4'd0; vp_win_padw = 4'd0;
    capture_win = 1'b1; win_ev_count = 0;

    vp_start_and_wait(VSRC_MEM_WIN, VDST_POOL, 24'h000001, 24'd0,
                      16'd0, 4'd0, 4'd0);
    capture_win = 1'b0;

    begin : max_buf_check
        reg       ok;
        integer   k;
        reg [7:0] exp [0:15];
        exp[0]=8'd1;  exp[1]=8'd2;  exp[2]=8'd4;  exp[3]=8'd5;
        exp[4]=8'd2;  exp[5]=8'd3;  exp[6]=8'd5;  exp[7]=8'd6;
        exp[8]=8'd4;  exp[9]=8'd5;  exp[10]=8'd7; exp[11]=8'd8;
        exp[12]=8'd5; exp[13]=8'd6; exp[14]=8'd8; exp[15]=8'd9;
        ok = 1'b1;
        if (win_ev_count !== 16) ok = 1'b0;
        for (k = 0; k < 16; k = k + 1)
        begin
            if (win_stream[k] !== exp[k]) ok = 1'b0;
            if (((k % 4) == 3) && (win_wend[k] !== 1'b1)) ok = 1'b0;
            if (((k % 4) != 3) && (win_wend[k] !== 1'b0)) ok = 1'b0;
        end
        check(ok === 1'b1, 21);
    end

    // -----------------------------------------------------------------------
    // TEST 22 (v0.7): move — contiguous device-memory copy main->main.
    //   Source main_mem[2..5] = 0x51,0x52,0x53,0x54; copy 4 elements to dest
    //   base 0x10 (main_mem[16..19]). The copy path reads each element then
    //   writes it (VSRC_MEM -> VDST_MEM); the source region stays intact.
    // -----------------------------------------------------------------------
    do_reset;
    main_mem[2] = 8'h51; main_mem[3] = 8'h52;
    main_mem[4] = 8'h53; main_mem[5] = 8'h54;
    main_mem[16] = 8'hFF; main_mem[17] = 8'hFF;
    main_mem[18] = 8'hFF; main_mem[19] = 8'hFF;

    vp_start_and_wait(VSRC_MEM, VDST_MEM, 24'h000002, 24'h000010, 16'd4, 4'd0, 4'd0);
    @(posedge clk); #1; // let the final memory write propagate

    check((main_mem[16] === 8'h51) && (main_mem[17] === 8'h52) &&
          (main_mem[18] === 8'h53) && (main_mem[19] === 8'h54) &&
          (main_mem[2]  === 8'h51) && (main_mem[5]  === 8'h54), 22);

    // -----------------------------------------------------------------------
    // TEST 23 (v0.7): move output staging main->0x1.
    //   A result tensor at main_mem[2..5] = 0x61..0x64 is moved into the
    //   isolated 0x1 I/O buffer (dest 0x0001) for readback; the dest walks the
    //   buffer offset (buf_mem[0..3]). A weight sentinel elsewhere in main
    //   memory (main_mem[100]) must be untouched by the staged write, proving
    //   the copy targets the buffer and does not spill into main memory
    //   (retires LeNet-5 "Bug 2").
    // -----------------------------------------------------------------------
    do_reset;
    main_mem[2] = 8'h61; main_mem[3] = 8'h62;
    main_mem[4] = 8'h63; main_mem[5] = 8'h64;
    main_mem[100] = 8'hA5;  // weight sentinel — must survive the staged write
    buf_mem[0] = 8'hFF; buf_mem[1] = 8'hFF;
    buf_mem[2] = 8'hFF; buf_mem[3] = 8'hFF;

    vp_start_and_wait(VSRC_MEM, VDST_MEM, 24'h000002, 24'h000001, 16'd4, 4'd0, 4'd0);
    @(posedge clk); #1;

    check((buf_mem[0] === 8'h61) && (buf_mem[1] === 8'h62) &&
          (buf_mem[2] === 8'h63) && (buf_mem[3] === 8'h64) &&
          (main_mem[100] === 8'hA5), 23);

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
