/* 
 * File: Systolic_Array_Input_Buffer.v
 * Author: Gabe Litteken
 * Date: 6/18/2026
 * 
 * Wrapper module for the scfifo megafunction that allows for variable buffer
 * depth at compile time. The template that this is based on is located in
 * unused/
 */
module Systolic_Array_Input_Buffer #(
	parameter DEPTH = 8
)(
	input clock,
	input [7:0] data,
	input rdreq,
	input sclr,
	input wrreq,
	output wire empty,
	output wire full,
	output wire [7:0] q

    // ---------- PARAMETERS ---------- //
	// ---------- END PARAMETERS ---------- //

	// ---------- DEBUG ---------- //
	// ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
scfifo #(
	.add_ram_output_register("ON"),
	.intended_device_family("Cyclone V"),
	.lpm_numwords(DEPTH),
	.lpm_showahead("OFF"),
	.lpm_type("scfifo"),
	.lpm_width(8),
	.lpm_widthu($clog2(DEPTH)), //automatically calculates widthu based on DEPTH
	.overflow_checking("ON"),
	.underflow_checking("ON"),
	.use_eab("ON")
) scfifo_component (
	.clock(clock),
	.data(data),
	.rdreq(rdreq),
	.sclr(sclr),
	.wrreq(wrreq),
	.empty(empty),
	.full(full),
	.q(q),
	.aclr(),
	.almost_empty(),
	.almost_full(),
	.eccstatus(),
	.usedw()
);
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule