/* 
 * File: ALU.v
 * Author: Gabe Litteken
 * Date: 6/15/2026
 * 
 * Insert module description here
 */
module ALU(
	
	input i_clk, //intentionally not connected
	input i_trst,
	
	//control signals
	input i_enable,
	input i_clear,
	output wire o_write,
	input [2:0] i_alu_op, 
	
	
	//data signals
	input [7:0] i_data_a,
	input [7:0] i_data_b,
	output reg [7:0] o_data

   // ---------- PARAMETERS ---------- //
	// ---------- END PARAMETERS ---------- //

	// ---------- DEBUG ---------- //
	// ---------- END DEBUG ---------- //

);

// ---------- PARAMETERS ---------- //
parameter integer REQUANT_SHIFT = 8; // Right-shift bits for requantization

parameter [2:0]
	ADD = 3'h0,
	NOOP = 3'h7;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
// The ALU implementation is combinational, so i_enable is passed through to o_write
// for writing to the vector buffer.

// Requantizes ALU output
function automatic signed [7:0] requantize(input signed [31:0] value);

	reg signed [31:0] val_signed;
	reg signed [31:0] val_shifted;
	begin
	val_signed  = $signed(value);
	val_shifted = val_signed >>> REQUANT_SHIFT;
	requantize = (val_shifted > $signed(32'd127))  ? 8'h7F :
                (val_shifted < $signed(-32'd128)) ? 8'h80 :
                val_shifted[7:0];
	end
endfunction


assign o_write = i_enable;

always @ (*)
begin
	if ((i_trst == 1'b0) | (i_clear == 1'b1) | (i_enable == 1'b0))
	begin
		o_data = 8'd45; //indicates a default state
	end
	else
	begin
		case (i_alu_op)
			ADD: o_data = requantize(i_data_a + i_data_b);
			NOOP: o_data = 8'd98; //indicates NOOP
			default: o_data = 8'd29; //indicates control error
		endcase 
	end
end
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
//Insert debug code here.
// ---------- END DEBUG ---------- //
endmodule