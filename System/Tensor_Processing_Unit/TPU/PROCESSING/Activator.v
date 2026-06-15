/* 
 * File: Activator.v
 * Author: Gabe Litteken
 * Date: 6/15/2026
 * 
 * Insert module description here
 */
module Activator(

	input i_clk,
	input i_trst,
	
	//control signals
	input i_enable,
	input i_clear,
	output wire o_write,
	input [2:0] i_activator_op,
	
	//data signals
	input [7:0] i_data,
	output wire [7:0] o_data
	
	// ---------- PARAMETERS ---------- //
	// ---------- END PARAMETERS ---------- //
	
   // ---------- DEBUG ---------- //
   // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
parameter [2:0]
	RELU = 3'h0;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
always @ (*)
begin
	case (i_activator_op)
		o_data =
	endcase 
end
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //



endmodule