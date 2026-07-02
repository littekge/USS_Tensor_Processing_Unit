/*
 * File: Vector_Processor.v
 * Author: Gabe Litteken
 * Date: 2026-06-15
 *
 * Handles memory accesses, data routing to and from the Activator, ALU, and
 * Systolic Array, and ISA-compliant formatting of data. Two instances are
 * used by the TPU: instance A (port a of memories, top SA buffers, Activator,
 * ALU input A, SA output read) and instance B (port b of memories, left SA
 * buffers, ALU input B).
 *
 * Memory abstraction: ISA address 0x0 maps to a hardwired-zero source,
 * 0x1 maps to TPU_0x1_Buffer (internal offset starting at 0), and
 * addresses >= 0x2 map to Weight_Memory at (ISA_addr - 2).
 *
 * SA loading convention (for C = rs1 * rs2, rs1 M x K, rs2 K x N):
 *   VP_A (VDST_SA_A): loads rows of rs1 into top SA input buffers
 *     top_buf[j] receives row j of rs1, j = 0..dim0-1, K = dim1 elements
 *   VP_B (VDST_SA_B): loads columns of rs2 into left SA input buffers
 *     left_buf[i] receives column i of rs2, i = 0..dim1-1, K = dim0 elements
 * SA readback (VSRC_SA_OUT): VP_A reads MAC(sa_row=col_cnt, sa_col=row_cnt)
 *   for output result[row_cnt][col_cnt], writing row-major to dest memory.
 *
 * Requantization: right-shifts 32-bit MAC accumulator by REQUANT_SHIFT bits
 * and saturates to the signed 8-bit range [-128, 127].
 */
module Vector_Processor (

    input              i_clk,
    input              i_trst,
    input              i_vector_start,

    // Controller combinational control inputs
    input  [2:0]       i_vect_source,
    input  [2:0]       i_vect_dest,
    input  [15:0]      i_mem_source_address,
    input  [15:0]      i_mem_dest_address,
    input  [15:0]      i_length,
    input  [3:0]       i_dim0,
    input  [3:0]       i_dim1,

    // ---------- PARAMETERS ---------- //
    // ---------- END PARAMETERS ---------- //

    // Handshake outputs to Controller
    output wire        o_vector_idle,
    output reg         o_element_valid,

    // Weight Memory port interface
    output reg  [15:0] o_wm_address,
    output reg         o_wm_wren,
    output reg  [7:0]  o_wm_data,
    input       [7:0]  i_wm_q,

    // TPU_0x1_Buffer port interface
    output reg  [15:0] o_buf_address,
    output reg         o_buf_wren,
    output reg  [7:0]  o_buf_data,
    input       [7:0]  i_buf_q,

    // Vector buffer interface (read only — writes come from Activator and ALU)
    output reg         o_vb_rdreq,
    input       [7:0]  i_vb_q,

    // Data output forwarded to Activator or ALU
    output reg  [7:0]  o_data,

    // Systolic array top input buffer interface (VP_A, VDST_SA_A)
    output reg  [7:0]  o_sa_top_data,
    output reg  [7:0]  o_sa_top_wrreq,   // one-hot, bit j = wrreq for top_buf[j]

    // Systolic array left input buffer interface (VP_B, VDST_SA_B)
    output reg  [7:0]  o_sa_left_data,
    output reg  [7:0]  o_sa_left_wrreq,  // one-hot, bit i = wrreq for left_buf[i]

    // Systolic array output read interface (VP_A, VSRC_SA_OUT)
    output reg  [3:0]  o_sa_row,
    output reg  [3:0]  o_sa_col,
    input       [31:0] i_sa_c

    // ---------- DEBUG ---------- //
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //

parameter integer REQUANT_SHIFT = 7; // Right-shift bits for requantization

// State machine states
parameter [3:0]
    STATE_ERROR  = 4'd0,
    START        = 4'd1,
    IDLE         = 4'd2,
    MEM_ADDR     = 4'd3,   // Assert memory read address
    MEM_WAIT     = 4'd4,   // Wait for registered RAM output (latency cycle 1)
    MEM_FORWARD  = 4'd5,   // RAM output valid; route to destination
    VB_RDREQ     = 4'd6,   // Assert vector buffer rdreq
    VB_WAIT      = 4'd7,   // Wait for FIFO output (latency cycle 1)
    VB_WRITE     = 4'd8,   // Write FIFO output to memory
    SA_OUT_SEL   = 4'd9,   // Present SA row/col selectors; c is combinational
    SA_OUT_WRITE = 4'd10,  // Write requantized MAC value to memory
    DONE_STATE   = 4'd11,
    MEM_ERROR    = 4'd12,
    VB_WAIT2     = 4'd13,  // Wait for FIFO output (latency cycle 2)
    MEM_WAIT2    = 4'd14;  // Wait for registered RAM output (latency cycle 2)

// Source encoding (must match Controller.v)
parameter [2:0]
    VSRC_MEM     = 3'd0,
    VSRC_SA_OUT  = 3'd1,
    VSRC_VEC_BUF = 3'd2;

// Destination encoding (must match Controller.v)
parameter [2:0]
    VDST_MEM   = 3'd0,
    VDST_SA_A  = 3'd1,
    VDST_SA_B  = 3'd2,
    VDST_ACT   = 3'd3,
    VDST_ALU_A = 3'd4,
    VDST_ALU_B = 3'd5;

// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// State machine registers
reg [3:0] S, NS;

// Latched control signals (stable throughout an operation)
reg [2:0]  vect_source_lat;
reg [2:0]  vect_dest_lat;
reg [15:0] mem_source_lat;   // ISA source base address
reg [15:0] mem_dest_lat;     // ISA dest base address
reg [15:0] length_lat;       // element count for length-based operations
reg [3:0]  dim0_lat;         // rows (SA ops) or outer dimension
reg [3:0]  dim1_lat;         // cols (SA ops) or inner dimension

// Element counters
reg [15:0] elem_cnt;   // flat element index (linear and VB operations)
reg [3:0]  row_cnt;    // row index for SA operations
reg [3:0]  col_cnt;    // column index for SA operations

// ISA source address for the current element.
// VDST_SA_B uses a stride: ISA_offset = row_cnt*dim1_lat + col_cnt.
// All other VSRC_MEM operations use a flat sequential offset.
wire [7:0]  sa_b_stride;   // zero-extended multiply to avoid 4-bit truncation
wire [15:0] cur_rd_isa;
assign sa_b_stride  = {4'd0, row_cnt} * {4'd0, dim1_lat};
assign cur_rd_isa   = (vect_dest_lat == VDST_SA_B)
                    ? (mem_source_lat + {8'd0, sa_b_stride} + {12'd0, col_cnt})
                    : (mem_source_lat + elem_cnt);

// ISA dest address for current write element (always sequential)
wire [15:0] cur_wr_isa;
assign cur_wr_isa = mem_dest_lat + elem_cnt;

// Source type decode (based on latched base address)
wire src_is_zero = (mem_source_lat == 16'h0000);
wire src_is_buf  = (mem_source_lat == 16'h0001);

// Destination type decode
wire dst_is_zero = (mem_dest_lat == 16'h0000);
wire dst_is_buf  = (mem_dest_lat == 16'h0001);

// Internal memory addresses (subtract the ISA base and reserved offsets).
// For 0x1 Buffer, the internal offset is the element index within the buffer:
//   buf_rd/wr_addr = cur_rd/wr_isa - mem_source/dest_lat = elem_cnt (linear)
// For Weight_Memory, the internal address is the ISA address minus 2 (reserved).
wire [15:0] wm_rd_addr;
wire [15:0] wm_wr_addr;
wire [15:0] buf_rd_addr;
wire [15:0] buf_wr_addr;
assign wm_rd_addr  = cur_rd_isa - 16'd2;
assign wm_wr_addr  = cur_wr_isa - 16'd2;
assign buf_rd_addr = cur_rd_isa - mem_source_lat;
assign buf_wr_addr = cur_wr_isa - mem_dest_lat;

// Muxed data read from memory based on source type
reg [7:0] mem_rd_data; // registered on posedge after MEM_WAIT
always @(*)
begin
    if (src_is_zero)
        mem_rd_data = 8'd0;
    else if (src_is_buf)
        mem_rd_data = i_buf_q;
    else
        mem_rd_data = i_wm_q;
end

// Requantization of 32-bit MAC accumulator to signed 8-bit with saturation
wire signed [31:0] sa_c_signed;
wire signed [31:0] sa_c_shifted;
assign sa_c_signed  = $signed(i_sa_c);
assign sa_c_shifted = sa_c_signed >>> REQUANT_SHIFT;

wire [7:0] sa_c_quantized;
assign sa_c_quantized = (sa_c_shifted > $signed(32'd127))  ? 8'h7F :
                        (sa_c_shifted < $signed(-32'd128)) ? 8'h80 :
                        sa_c_shifted[7:0];

// Lookahead done flags: evaluate CURRENT counter values to decide whether
// the element being processed in the current cycle is the last one, so the
// state machine can transition to DONE_STATE after the next posedge.
wire sa_a_last   = (row_cnt == dim0_lat - 4'd1) && (col_cnt == dim1_lat - 4'd1);
wire sa_b_last   = (col_cnt == dim1_lat - 4'd1) && (row_cnt == dim0_lat - 4'd1);
wire sa_out_last = (row_cnt == dim0_lat - 4'd1) && (col_cnt == dim1_lat - 4'd1);
wire linear_last = (elem_cnt + 16'd1 >= length_lat);

// State machine driver
always @(posedge i_clk or negedge i_trst)
begin
    if (i_trst == 1'b0)
        S <= START;
    else
        S <= NS;
end

// Next state logic
always @(*)
begin
    case (S)
        START:
            NS = IDLE;
        IDLE:
        begin
            if (i_vector_start)
            begin
                // Dispatch directly to the correct first operational state
                if (i_vect_source == VSRC_SA_OUT)
                    NS = SA_OUT_SEL;
                else if (i_vect_source == VSRC_VEC_BUF)
                    NS = VB_RDREQ;
                else
                    NS = MEM_ADDR;   // VSRC_MEM
            end
            else
                NS = IDLE;
        end
        MEM_ADDR:
            // Detect illegal write target before committing to a memory wait.
            // The reserved-address (0x0) write check only applies when the
            // destination is actually device memory; for systolic-array,
            // activator, and ALU destinations the dest address is unused, so a
            // zero value there is legal (the Controller leaves it at 0).
            NS = (vect_dest_lat == VDST_MEM && dst_is_zero) ? MEM_ERROR : MEM_WAIT;
        MEM_WAIT:
            // Weight_Memory / 0x1 Buffer (registered address + registered
            // output) have a two-cycle read latency, so a second wait state is
            // required before the RAM output is valid on i_wm_q / i_buf_q.
            NS = MEM_WAIT2;
        MEM_WAIT2:
            NS = MEM_FORWARD;
        MEM_FORWARD:
        begin
            if (vect_dest_lat == VDST_SA_A)
                NS = sa_a_last ? DONE_STATE : MEM_ADDR;
            else if (vect_dest_lat == VDST_SA_B)
                NS = sa_b_last ? DONE_STATE : MEM_ADDR;
            else
                NS = linear_last ? DONE_STATE : MEM_ADDR;
        end
        VB_RDREQ:
            NS = VB_WAIT;
        VB_WAIT:
            // The Vector_Buffer scfifo (showahead OFF + output register ON) has
            // a two-cycle read latency, so a second wait state is required
            // before the popped element is valid on i_vb_q.
            NS = VB_WAIT2;
        VB_WAIT2:
            NS = VB_WRITE;
        VB_WRITE:
            NS = linear_last ? DONE_STATE : VB_RDREQ;
        SA_OUT_SEL:
            NS = SA_OUT_WRITE;
        SA_OUT_WRITE:
            NS = sa_out_last ? DONE_STATE : SA_OUT_SEL;
        DONE_STATE:
            NS = IDLE;
        MEM_ERROR:
            NS = MEM_ERROR;
        STATE_ERROR:
            NS = STATE_ERROR;
        default:
            NS = STATE_ERROR;
    endcase
end

// Execution logic
always @(posedge i_clk or negedge i_trst)
begin
    if (i_trst == 1'b0)
    begin
        vect_source_lat  <= 3'd0;
        vect_dest_lat    <= 3'd0;
        mem_source_lat   <= 16'd0;
        mem_dest_lat     <= 16'd0;
        length_lat       <= 16'd0;
        dim0_lat         <= 4'd0;
        dim1_lat         <= 4'd0;
        elem_cnt         <= 16'd0;
        row_cnt          <= 4'd0;
        col_cnt          <= 4'd0;
        o_wm_address     <= 16'd0;
        o_wm_wren        <= 1'b0;
        o_wm_data        <= 8'd0;
        o_buf_address    <= 16'd0;
        o_buf_wren       <= 1'b0;
        o_buf_data       <= 8'd0;
        o_vb_rdreq       <= 1'b0;
        o_data           <= 8'd0;
        o_sa_top_data    <= 8'd0;
        o_sa_top_wrreq   <= 8'd0;
        o_sa_left_data   <= 8'd0;
        o_sa_left_wrreq  <= 8'd0;
        o_sa_row         <= 4'd0;
        o_sa_col         <= 4'd0;
        o_element_valid  <= 1'b0;
    end
    else
    begin
        // Default: deassert all one-cycle signals
        o_wm_wren       <= 1'b0;
        o_buf_wren      <= 1'b0;
        o_vb_rdreq      <= 1'b0;
        o_sa_top_wrreq  <= 8'd0;
        o_sa_left_wrreq <= 8'd0;
        o_element_valid <= 1'b0;

        case (S)

            IDLE:
            begin
                if (i_vector_start)
                begin
                    // Latch control inputs and reset counters
                    vect_source_lat <= i_vect_source;
                    vect_dest_lat   <= i_vect_dest;
                    mem_source_lat  <= i_mem_source_address;
                    mem_dest_lat    <= i_mem_dest_address;
                    length_lat      <= i_length;
                    dim0_lat        <= i_dim0;
                    dim1_lat        <= i_dim1;
                    elem_cnt        <= 16'd0;
                    row_cnt         <= 4'd0;
                    col_cnt         <= 4'd0;
                end
            end

            MEM_ADDR:
            begin
                // Drive the correct memory read address for the current element.
                // The mux on cur_rd_isa_addr handles SA_B stride vs linear.
                if (src_is_buf)
                begin
                    o_buf_address <= buf_rd_addr;
                end
                else if (!src_is_zero)
                begin
                    o_wm_address <= wm_rd_addr;
                end
                // Zero source: no address needed, mem_rd_data returns 0 combinationally
            end

            MEM_WAIT:;  // RAM read latency cycle 1; hold address
            MEM_WAIT2:; // RAM read latency cycle 2; i_wm_q/i_buf_q valid next cycle

            MEM_FORWARD:
            begin
                // mem_rd_data is now valid. Route to destination.
                case (vect_dest_lat)
                    VDST_ACT, VDST_ALU_A, VDST_ALU_B:
                    begin
                        // Both o_data and o_element_valid are registered (NBA).
                        // Setting o_element_valid here means it is HIGH on the
                        // NEXT cycle, which is exactly when o_data is also valid.
                        o_data          <= mem_rd_data;
                        o_element_valid <= 1'b1;
                    end

                    VDST_SA_A:
                    begin
                        // Push data to top SA input buffer for current row
                        o_sa_top_data          <= mem_rd_data;
                        o_sa_top_wrreq         <= (8'd1 << row_cnt);

                        // Advance col counter; when a row is complete advance row
                        col_cnt <= col_cnt + 4'd1;
                        if (col_cnt + 4'd1 == dim1_lat)
                        begin
                            col_cnt <= 4'd0;
                            row_cnt <= row_cnt + 4'd1;
                            elem_cnt <= elem_cnt + 16'd1;
                        end
                        else
                        begin
                            elem_cnt <= elem_cnt + 16'd1;
                        end
                    end

                    VDST_SA_B:
                    begin
                        // Push data to left SA input buffer for current column
                        o_sa_left_data          <= mem_rd_data;
                        o_sa_left_wrreq         <= (8'd1 << col_cnt);

                        // Advance row counter; when column is complete advance col
                        row_cnt <= row_cnt + 4'd1;
                        if (row_cnt + 4'd1 == dim0_lat)
                        begin
                            row_cnt <= 4'd0;
                            col_cnt <= col_cnt + 4'd1;
                            elem_cnt <= elem_cnt + 16'd1;
                        end
                        else
                        begin
                            elem_cnt <= elem_cnt + 16'd1;
                        end
                    end

                    default:; // Other destinations not valid with VSRC_MEM→write
                endcase

                // For linear (non-SA) ops, increment elem_cnt
                if (vect_dest_lat != VDST_SA_A && vect_dest_lat != VDST_SA_B)
                    elem_cnt <= elem_cnt + 16'd1;
            end

            VB_RDREQ:
            begin
                // Assert rdreq to pop one byte from the vector buffer
                o_vb_rdreq <= 1'b1;
            end

            VB_WAIT:;  // FIFO read latency cycle 1
            VB_WAIT2:; // FIFO read latency cycle 2; i_vb_q valid next cycle

            VB_WRITE:
            begin
                // Write the FIFO output to destination memory
                if (dst_is_buf)
                begin
                    o_buf_address <= buf_wr_addr;
                    o_buf_data    <= i_vb_q;
                    o_buf_wren    <= 1'b1;
                end
                else
                begin
                    // Destination >= 0x2: Weight_Memory
                    o_wm_address <= wm_wr_addr;
                    o_wm_data    <= i_vb_q;
                    o_wm_wren    <= 1'b1;
                end
                elem_cnt <= elem_cnt + 16'd1;
            end

            SA_OUT_SEL:
            begin
                // Select which MAC output to read. Reading MAC(row=col_cnt, col=row_cnt)
                // places result[row_cnt][col_cnt] = (rs1*rs2)[row_cnt][col_cnt].
                o_sa_row <= col_cnt;
                o_sa_col <= row_cnt;
            end

            SA_OUT_WRITE:
            begin
                // Write requantized MAC value to destination memory
                if (dst_is_buf)
                begin
                    o_buf_address <= buf_wr_addr;
                    o_buf_data    <= sa_c_quantized;
                    o_buf_wren    <= 1'b1;
                end
                else
                begin
                    o_wm_address <= wm_wr_addr;
                    o_wm_data    <= sa_c_quantized;
                    o_wm_wren    <= 1'b1;
                end

                // Advance col counter; when all cols done advance row
                col_cnt <= col_cnt + 4'd1;
                if (col_cnt + 4'd1 == dim1_lat)
                begin
                    col_cnt  <= 4'd0;
                    row_cnt  <= row_cnt + 4'd1;
                end
                elem_cnt <= elem_cnt + 16'd1;
            end

            default:;
        endcase
    end
end

// o_vector_idle: HIGH when the VP is idle (waiting for a new operation)
assign o_vector_idle = (S == START || S == IDLE);

// o_element_valid: HIGH for exactly one cycle after MEM_FORWARD when a valid
// element is being forwarded to the Activator or ALU. Registered (not a wire)
// so it is asserted on the cycle following MEM_FORWARD, precisely when o_data
// (also updated via non-blocking assignment in MEM_FORWARD) holds valid data.

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
