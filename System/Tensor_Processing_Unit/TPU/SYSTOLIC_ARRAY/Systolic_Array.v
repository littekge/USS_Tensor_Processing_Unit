/*
 * File: Systolic_Array.v
 * Author: Gabe Litteken
 * Date: 6/18/2026
 *
 * Top level systolic array module. Generates the systolic array using generate
 * blocks that instantiate and connect the multiply accumulate units and
 * systolic array input buffers. Also implements a state machine that loads
 * data to the systolic array from the input buffers in the correct order.
 *
 * Tiling convention (matches Vector_Processor data routing):
 *   - The first net index is the COLUMN (a/top stream), the second is the
 *     ROW (b/left stream). MAC(col=i, row=j) accumulates result[i][j].
 *   - a_in enters the top of a column, a_out leaves the bottom.
 *   - b_in enters the left of a row, b_out leaves the right.
 *   - top_buf[i] feeds the a_in of column i; left_buf[j] feeds the b_in of
 *     row j.
 *
 * Output-stationary skew: buffer M (top and left) begins streaming on the Mth
 * clock cycle after a start, so the a/b operands for a given contraction index
 * arrive at each MAC on the same cycle.
 */
module Systolic_Array #(parameter N = 8) (

	input i_clk,
	input i_trst,

	//synchronous control signals
	input i_clear,
	input i_systolic_array_start,
	output wire o_systolic_array_idle,

	//systolic array top input buffer signals
	input [7:0] i_top_data,
	input [N-1:0]  i_top_wrreq,   // one-hot, bit j = wrreq for top_buf[j]

	//systolic array left input buffer signals
	input [7:0] i_left_data,
	input [N-1:0] i_left_wrreq,  // one-hot, bit i = wrreq for left_buf[i]

	//systolic array output select and data
   input [3:0] i_row,
   input [3:0] i_col,
	output wire [31:0] o_c

	// ---------- PARAMETERS ---------- //
	// ---------- END PARAMETERS ---------- //

	// ---------- DEBUG ---------- //
	// ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
parameter [1:0]
	STATE_ERROR = 2'd0,
	START       = 2'd1,
	IDLE        = 2'd2,
	LOAD        = 2'd3;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

genvar i;
genvar j;

//defining buffer clear signal: flush on trst LOW or clear HIGH
wire buf_sclr;
assign buf_sclr = ~i_trst | i_clear;

//connection nets
//First index = column (a/top stream), second index = row (b/left stream).
//v_interconnect carries the a stream down a column: index [col][0..N], where
//row 0 is the top input and row N is the (unused) bottom output.
//h_interconnect carries the b stream across a row: index [0..N][row], where
//column 0 is the left input and column N is the (unused) right output.
wire [7:0] v_interconnect[0:N-1][0:N];
wire [7:0] h_interconnect[0:N][0:N-1];
wire [31:0] unit_out[0:N-1][0:N-1];

//per-buffer read requests and empty flags (one bit per buffer)
reg  [N-1:0] left_rdreq;
reg  [N-1:0] top_rdreq;
wire [N-1:0] left_empty;
wire [N-1:0] top_empty;

//raw buffer outputs (before the input-valid gate is applied)
wire [7:0] top_q[0:N-1];
wire [7:0] left_q[0:N-1];

//read-request valids delayed by one cycle to match the input-buffer read
//latency (rdreq at cycle T -> q valid at cycle T+1). A buffer drives genuine
//popped data into the array only while its delayed read request is HIGH; at all
//other times the array edge is fed 0. Because each MAC accumulates a*b every
//cycle unconditionally, feeding 0 once a buffer has drained makes the stale
//held-output contribute nothing, while the real operands already streamed in
//propagate through normally. This is what stops an output-stationary array from
//over-accumulating after its input buffers empty.
reg [N-1:0] top_rdreq_d;
reg [N-1:0] left_rdreq_d;

//ramp counter; in LOAD, buffers 0..load_cnt are enabled to read
reg [7:0] load_cnt;

//HIGH once every input buffer has been fully drained
wire all_empty;
assign all_empty = (&top_empty) & (&left_empty);

//decodes output: o_c selects MAC at (column=i_col, row=i_row)
assign o_c = unit_out[i_col][i_row];

//generates the left side input buffers (feed b_in of each row)
generate
	for (j = 0; j < N; j = j + 1)
	begin : gen_left_buf
		Systolic_Array_Input_Buffer #(.DEPTH(N)) SAIB_left (
			.clock(i_clk),
			.data(i_left_data),
			.rdreq(left_rdreq[j]),
			.sclr(buf_sclr),
			.wrreq(i_left_wrreq[j]),
			.empty(left_empty[j]),
			.full(),
			.q(left_q[j])
		);
		//gate the left edge of row j: real popped data only while streaming
		assign h_interconnect[0][j] = left_rdreq_d[j] ? left_q[j] : 8'd0;
	end
endgenerate

//generates the top side input buffers (feed a_in of each column)
generate
	for (i = 0; i < N; i = i + 1)
	begin : gen_top_buf
		Systolic_Array_Input_Buffer #(.DEPTH(N)) SAIB_top(
			.clock(i_clk),
			.data(i_top_data),
			.rdreq(top_rdreq[i]),
			.sclr(buf_sclr),
			.wrreq(i_top_wrreq[i]),
			.empty(top_empty[i]),
			.full(),
			.q(top_q[i])
		);
		//gate the top edge of column i: real popped data only while streaming
		assign v_interconnect[i][0] = top_rdreq_d[i] ? top_q[i] : 8'd0;
	end
endgenerate

//generates the systolic array (a flows down columns, b flows across rows)
generate
	for (i = 0; i < N; i = i + 1)
	begin : gen_mac_col
		for (j = 0; j < N; j = j + 1)
		begin : gen_mac_row
			Multiply_Accumulate_Unit MAC_i_j(
				.i_clk(i_clk),
				.i_trst(i_trst),
				.i_clear(i_clear),
				.i_a(v_interconnect[i][j]),
				.i_b(h_interconnect[i][j]),
				.o_a(v_interconnect[i][j+1]),
				.o_b(h_interconnect[i+1][j]),
				.o_c(unit_out[i][j])
			);
		end
	end
endgenerate


//---- State Machine ----

reg [1:0] S, NS;

//state machine driver
always @ (posedge i_clk or negedge i_trst)
begin
	if (i_trst == 1'b0)
	begin
		S <= START;
	end
	else
	begin
		S <= NS;
	end
end

//next state behavior
always @ (*)
begin
	case (S)
		START: NS = IDLE;
		IDLE:  NS = (i_systolic_array_start == 1'b1) ? LOAD : IDLE;
		LOAD:  NS = (all_empty == 1'b1) ? IDLE : LOAD;
		STATE_ERROR: NS = STATE_ERROR;
		default: NS = STATE_ERROR;
	endcase
end

//execution logic: drives the ramp counter that staggers buffer reads
always @ (posedge i_clk or negedge i_trst)
begin
	if (i_trst == 1'b0)
	begin
		load_cnt <= 8'd0;
	end
	else
	begin
		case (S)
			IDLE:
			begin
				load_cnt <= 8'd0; //hold ramp at 0 until loading begins
			end
			LOAD:
			begin
				//advance the ramp until every buffer is enabled
				if (load_cnt < (N - 1))
				begin
					load_cnt <= load_cnt + 8'd1;
				end
			end
			default:;
		endcase
	end
end

//delay the read-request valids by one cycle to align with the input-buffer
//read latency, so the input-valid gate matches when q actually holds popped
//data. Cleared with the array (trst LOW or clear HIGH) so each operation starts
//with no spurious valids.
always @(posedge i_clk or negedge i_trst)
begin
	if (i_trst == 1'b0)
	begin
		top_rdreq_d  <= {N{1'b0}};
		left_rdreq_d <= {N{1'b0}};
	end
	else if (i_clear == 1'b1)
	begin
		top_rdreq_d  <= {N{1'b0}};
		left_rdreq_d <= {N{1'b0}};
	end
	else
	begin
		top_rdreq_d  <= top_rdreq;
		left_rdreq_d <= left_rdreq;
	end
end

//read request decode: in LOAD, buffers 0..load_cnt read while not empty. The
//ramp on load_cnt produces the diagonal offset required by an output-stationary
//systolic array.
integer k;
always @ (*)
begin
	top_rdreq  = {N{1'b0}};
	left_rdreq = {N{1'b0}};
	if (S == LOAD)
	begin
		for (k = 0; k < N; k = k + 1)
		begin
			if (k <= load_cnt)
			begin
				top_rdreq[k]  = ~top_empty[k];
				left_rdreq[k] = ~left_empty[k];
			end
		end
	end
end

//idle signal set combinationally (idle while not loading)
assign o_systolic_array_idle = (S == START) || (S == IDLE);
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
