/* 
 * File: ALU.v
 * Author: Gabe Litteken
 * Date: 6/15/2026
 * 
 * Insert module description here
 */
module ALU(
	
	//control signals
	input [2:0] i_alu_op, 
	input i_enable
	output o_buffer_write
	
	//data signals
	input [7:0] i_data_a,
	input [7:0] i_data_b,
	output wire [7:0] o_data
	
	

   // ---------- PARAMETERS ---------- //
	// ---------- END PARAMETERS ---------- //

	// ---------- DEBUG ---------- //
	// ---------- END DEBUG ---------- //

);

// ---------- PARAMETERS ---------- //
parameter [2:0]
	ADD = 3'h0;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
assign o_buffer_write = i_enable;

always @ (*)
begin
	case (i_alu_op)
		ADD: 
	
	
	endcase
end	

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
//Insert debug code here.
// ---------- END DEBUG ---------- //
endmodule