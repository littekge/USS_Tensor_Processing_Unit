/* 
 * File: ascii_master_controller.v
 * Author: Gabe Litteken
 * Date: 5/19/2026
 * 
 * This module is an interface between a RAM that stores ASCII characters and 
 * color values and a VGA controller. Characters are arraneged in a 60x80 grid.
 */
module ascii_master_controller (

	input clk, //input clock
	input rst, //reset signal
	
	input ascii_write_en,  //enables writing to the master buffer
	input [31:0]ascii_input, //master buffer data
	input [12:0]ascii_write_address, //master buffer write address
	
	//VGA signal passthrough
	output wire vga_blank,
	output wire [7:0]vga_b,
	output wire	[7:0]vga_r,
	output wire	[7:0]vga_g,
	output wire	vga_clk,
	output wire	vga_hs,
	output wire	vga_vs,
	output wire	vga_sync

	// ---------- DEBUG ---------- //
	//input [9:0]SW,
	//input [3:0]KEY,
	//output [9:0]LEDR
	// ---------- END DEBUG ---------- //
);


// ---------- CODE ---------- //
wire [12:0] vga_read_address; //read address from the vga controller
wire [31:0] vga_data; //data for the above read address

//vga controller declaration
vga_controller controller(
	.clk(clk), //input clock
	.rst(rst), //reset signal
	
	.ascii_buffer_address(vga_read_address), //output address from the vga controller
	.ascii_buffer_data(vga_data), //input data for the vga controller
	
	//VGA signal passthrough
	.vga_blank(vga_blank),
	.vga_b(vga_b),
	.vga_r(vga_r),
	.vga_g(vga_g),
	.vga_clk(vga_clk),
	.vga_hs(vga_hs),
	.vga_vs(vga_vs),
	.vga_sync(vga_sync)
	
	/* ---------- DEBUG ---------- */
	//.SW(SW[9:0]),
	//.KEY(KEY[3:0]),
	//.LEDR(LEDR[9:0])
	/* ---------- END DEBUG ---------- */
);

//Master RAM declaration
wire [12:0]master_read_address;
wire [12:0]master_write_address;
wire [31:0]master_data_out;
wire [31:0]master_data_in;
wire master_write_en;

ascii_master master (
	.rdaddress(master_read_address),
	.wraddress(master_write_address),
	.clock(clk),
	.data(master_data_in),
	.wren(master_write_en),
	.q(master_data_out)
);

//Signal assignment
assign master_read_address = vga_read_address; //read address assigned to address given by vga controller
assign master_write_en = ascii_write_en; //passing through enable signal
assign master_data_in = ascii_input; //passing through ASCII input data
assign master_write_address = ascii_write_address; //passing trhough write address
assign vga_data = master_data_out; //passing VGA data to VGA controller

// ---------- END CODE ---------- //



// ---------- DEBUG ---------- //
//wire [3:0]DELAY_TIME;

//assign DELAY_TIME = SW[3:0];

//wire test_clk;
//assign test_clk = KEY[3];

//assign LEDR[2:0] = WRITE_S;
// ---------- END DEBUG ---------- //

endmodule