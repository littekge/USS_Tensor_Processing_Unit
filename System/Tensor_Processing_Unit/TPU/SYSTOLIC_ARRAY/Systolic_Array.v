/* 
 * File: Systolic_Array.v
 * Author: Gabe Litteken
 * Date: 6/18/2026
 * 
 * Top level systolic array module. Generates the systolic array using generate
 * blocks that instantiate and connect the multiply accumulate units and
 * systolic array input buffers. Also implements a state machine that loads
 * data to the systolic array from the input buffers in the correct order.
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
parameter [2:0]
	STATE_ERROR = 3'd0,
	START = 3'd1,
	IDLE = 3'd2,
	LOAD_FOR_CHECK = 3'd3,
	LOAD_FOR_EXECUTE = 3'd4,
	LOAD_FOR_INCREMENT = 3'd5;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

genvar i;
genvar j;

//defining buffer clear signal
wire buf_sclr;
assign buf_sclr = ~i_trst | i_clear;

//connection nets
//NOTE: consider the first dimension of each net the horizontal dimension and
//the second dimension the vertical dimension.
wire [7:0] h_interconnect[0:N-1][0:N-1];
wire [7:0] v_interconnect[0:N-1][0:N-1];
wire [31:0] unit_out[0:N-1][0:N-1];
wire left_rdreq[0:N-1];
wire top_rdreq[0:N-1];
wire all_empty;

//decodes output
assign o_c = unit_out[i_col][i_row];

//generates the left side input buffers
generate
	for (j = 0; j < N; j = j + 1)
	begin
		Systolic_Array_Input_Buffer #(.DEPTH(N)) SAIB_left (
			.clock(i_clk),
			.data(i_left_data),
			.rdreq(left_rdreq[j]),
			.sclr(buf_sclr),
			.wrreq(i_left_wrreq[j]),
			.empty(all_empty),
			.full(),
			.q(h_interconnect[0][j])
		);
	end
endgenerate

//generates the top side input buffers
generate
	for (i = 0; i < N; i = i + 1)
	begin
		Systolic_Array_Input_Buffer #(.DEPTH(N)) SAIB_top(
			.clock(i_clk),
			.data(i_top_data),
			.rdreq(top_rdreq[i]),
			.sclr(buf_sclr),
			.wrreq(i_top_wrreq[i]),
			.empty(all_empty),
			.full(),
			.q(v_interconnect[i][0])
		);
	end
endgenerate

//generates the systolic array
generate
	for (i = 0; i < N; i = i + 1)
	begin
		for (j = 0; j < N; j = j + 1)
		begin
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

reg [2:0] S, NS;

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
		IDLE: NS = (i_systolic_array_start == 1'b1)?LOAD_WHILE_CHECK:IDLE;
		LOAD_WHILE_CHECK:;
		STATE_ERROR: NS = STATE_ERROR;
		default: NS = STATE_ERROR;
	endcase
end

//execution logic
always @ (posedge i_clk or negedge i_trst)
begin
	if (i_trst == 1'b0)
	begin
	
	end
	else
	begin
		case (S)
			default:;
		endcase
	end
end

assign o_systolic_array_idle = (S == IDLE)?1'b1:1'b0; //idle signal set combinationally
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule