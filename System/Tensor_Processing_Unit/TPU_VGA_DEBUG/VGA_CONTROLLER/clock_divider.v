/* 
 * File: clock_divider.v
 * Author: Gabe Litteken
 * Date: 5/19/2026
 *
 * Simple clock divider circuit
 */
module clock_divider(

	input clk, //input clock
	input rst, //reset signal
	output reg clk_25 //divided output clock
	
	);
	

	// ---------- CODE ---------- //
	always @ (posedge clk or negedge rst)
	begin
		if (rst == 1'b0)
			clk_25 <= 1'b0;
		else 
			clk_25 <= ~clk_25;
	end
	// ---------- END CODE ---------- //
endmodule