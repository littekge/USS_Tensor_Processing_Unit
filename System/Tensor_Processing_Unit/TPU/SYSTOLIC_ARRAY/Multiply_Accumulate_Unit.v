/* 
 * File: Multiply_Accumulate_Unit.v
 * Author: Gabe Litteken
 * Date: 6/17/2026
 * 
 * Serves as the basic building block of a systolic array. Every clock cycle
 * the module computes a multiply accumulate operation c = c + a * b and
 * forwards a and b to the next tiles in the array.
 */
module Multiply_Accumulate_Unit (

   input i_clk,
	input i_trst,
	
	//control
	input i_clear,
	
	//data
	input [7:0] i_a,
	input [7:0] i_b,
	output reg [7:0] o_a,
	output reg [7:0] o_b,
	output reg [31:0] o_c

	// ---------- PARAMETERS ---------- //
	// ---------- END PARAMETERS ---------- //

	// ---------- DEBUG ---------- //
	// ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
always @ (posedge i_clk or negedge i_trst)
begin
	if ((i_trst == 1'b0) || (i_clear == 1'b1))
	begin
		o_a <= 8'd0;
		o_b <= 8'd0;
		o_c <= 32'd0;
	end
	else
	begin
		o_a <= i_a;
		o_b <= i_b;
		o_c <= o_c + (i_a * i_b);
	end
end
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule