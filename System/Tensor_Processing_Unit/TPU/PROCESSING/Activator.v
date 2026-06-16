/* 
 * File: Activator.v
 * Author: Gabe Litteken
 * Date: 6/15/2026
 * 
 * Combinational module to implement common activation functions used in
 * neural network processing. 
 */
module Activator(

	input i_clk, //intentionally not connected
	input i_trst,
	
	//control signals
	input i_enable,
	input i_clear,
	output wire o_write,
	input [2:0] i_activator_op,
	
	//data signals
	input [7:0] i_data,
	output reg [7:0] o_data
	
	// ---------- PARAMETERS ---------- //
	// ---------- END PARAMETERS ---------- //
	
   // ---------- DEBUG ---------- //
   // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
parameter [2:0]
	RELU = 3'h0,
	NOOP = 3'h7;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
// The Activator implementation is combinational, so i_enable is passed through to o_write
// for writing to the vector buffer.
assign o_write = i_enable;

always @ (*)
begin
	if ((i_trst == 1'b0) | (i_clear == 1'b1) | (i_enable == 1'b0))
	begin
		o_data = 8'd45; //indicates a default state
	end
	else
	begin
		case (i_activator_op)
			RELU: o_data = ($signed(i_data) > $signed(8'd0))?i_data:8'd0;
			NOOP: o_data = 8'd98; //indicates NOOP
			default: o_data = 8'd29; //indicates control error
		endcase 
	end
end
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //

endmodule