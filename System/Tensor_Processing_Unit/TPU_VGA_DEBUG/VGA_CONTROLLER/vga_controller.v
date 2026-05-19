/* 
 * File: vga_controller.v
 * Author: Gabe Litteken
 * Date: 5/19/2026
 * 
 * This module is a simple VGA controller designed to read ASCII codes and
 * colors from a master memory, and translate them into pixels for the VGA
 * driver to display.
 */

module vga_controller (

	input clk, //input clock
	input rst, //reset signal

	output reg [12:0] ascii_buffer_address, //address to retrieve data from the master buffer
	input [31:0] ascii_buffer_data, //returned data from master buffer
	
	//VGA signal passthrough
	output wire vga_blank,
	output wire [7:0]vga_b,
	output wire	[7:0]vga_r,
	output wire	[7:0]vga_g,
	output wire	vga_clk,
	output wire	vga_hs,
	output wire	vga_vs,
	output wire	vga_sync,
	
	// ---------- DEBUG ---------- //
	input [9:0]SW,
	input [3:0]KEY,
	input [9:0]LEDR
	// ---------- END DEBUG ---------- //
	
);



// ---------- CODE ---------- //
wire clk_25; //divided clock for VGA

wire [9:0] next_x, next_y; //coordinates of the next pixel on the screen.
reg [7:0] red, green, blue; //color signals

reg [9:0]ascii_rom_address; //address to decode ASCII into pixels
reg current_pixel; //current pixel value
wire [7:0]ascii_rom_data; //data returned from the ASCII ROM

clock_divider clock(
	.clk(clk), //input clock
	.rst(rst), //reset signal
	.clk_25(clk_25) //25 MHz output clock
);


vga_driver driver(

	.clk_25(clk_25), //input clock
	.rst(rst), //reset signal
	
	//inputs
	.r(red),
	.g(green),
	.b(blue),
	
	//control outputs
	.x(next_x),
	.y(next_y),
	.disp_done(),
	
	//outputs
	.vga_blank(vga_blank),
	.vga_b(vga_b),
	.vga_r(vga_r),
	.vga_g(vga_g),
	.vga_clk(vga_clk),
	.vga_hs(vga_hs),
	.vga_vs(vga_vs),
	.vga_sync_n(vga_sync),
	
	
	// ---------- DEBUG ---------- //
	.SW(SW),
	.KEY(KEY),
	.LEDR(LEDR[9:0])
	// ---------- END DEBUG ---------- //
);

Char_ROM rom1 (
	.address(ascii_rom_address),
	.clock(clk),
	.q(ascii_rom_data)
);

//combinational ram control
always @ (*)
begin
	ascii_buffer_address = 80 * (next_y >> 3) + (next_x >> 3); //decodes next X and Y coordinates into an address in the master buffer
	
	ascii_rom_address = (ascii_buffer_data[31:24] * 8) + (next_y % 8); //decodes ASCII code and Y coordinate into a ROM address to retrieve a specific line of a character
	current_pixel = ((ascii_rom_data >> ((next_x - 10'd2) % 8)) & 1'b1); //determines whether a pixel is on or off based on X coordinate
	
	if (current_pixel == 1'b1)
		{red, green, blue} = ascii_buffer_data[23:0];
	else
		{red, green, blue} = 24'b0;
		
end
// ---------- END CODE ---------- //



// ---------- DEBUG ---------- //
/*
always @ (*)
begin
	red = 8'b11111111;
	blue = 8'b11111111;
	green = 8'b11111111;
end
*/


/*
reg [9:0]ROM_OFFSET;

always @ (*)
begin
	ROM_OFFSET = SW[5:3];
end
*/
// ---------- END DEBUG ---------- //
endmodule
