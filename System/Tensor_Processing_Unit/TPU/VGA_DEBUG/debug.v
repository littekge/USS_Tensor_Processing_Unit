/* 
 * File: debug.v
 * Author: Gabe Litteken
 * Date: 5/19/2026
 * 
 * This module is an interface between the TPU and the ASCII VGA controller.
 * It's designed to translate signed numbers to hex codes and output them to
 * a monitor for debugging purposes.
 */
module debug (

   input i_clk, //input clock
	input i_rst, //reset signal
	
	input [31:0]i_data, //signed integer to write to the screen
	input i_write_next, //pulse high for one clock cycle to write the integer in i_data
	output reg o_write_ready, //high if the module is ready for the next integer

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

// ---------- PARAMETERS ---------- //
//state machine states
parameter 
	ERROR = 3'd0,
	START = 3'd1,
	WAIT_WREN = 3'd2,
	COL_CHECK = 3'd3,
	WRITE = 3'd4,
	COL_INC = 3'd5,
	ROW_INC = 3'd6;
// ---------- END PARAMETERS ---------- //


// ---------- CODE ---------- //
reg [3:0] col_num; //column number
reg [7:0] row_num; //row number

reg [31:0] data; //register to store input data

//State machine variables
reg [2:0]S, NS;

//ASCII VGA Controller variables
reg [7:0]ascii_code;
reg ascii_wren;
wire [31:0]ascii_data;
wire [12:0]ascii_addr;

//State machine driver
always @ (posedge i_clk or negedge i_rst)
begin
	if (i_rst == 1'b0)
	begin
		S <= START; //sets state to START if reset is triggered
	end
	else
	begin
		S <= NS; //otherwise set S to NS
	end
end

//Determining next state
always @ (*)
begin
	case (S)
		START: NS = WAIT_WREN;
		WAIT_WREN: NS = (i_write_next == 1'b1)?(COL_CHECK):(WAIT_WREN);  //idles in WAIT_WREN until write next signal is recieved
		COL_CHECK: NS = (col_num < 4'd9)?(WRITE):(ROW_INC); //operates a for loop 
		WRITE: NS = COL_INC;
		COL_INC: NS = COL_CHECK;
		ROW_INC: NS = WAIT_WREN;
		default: NS = ERROR;
	endcase
end

//Execution logic
always @ (posedge i_clk or negedge i_rst)
begin
	if (i_rst == 1'b0)
	begin
		//zeroing registers on reset
		row_num <= 8'd0;
		col_num <= 4'd0;
		ascii_wren <= 1'd0;
		o_write_ready <= 1'd0;
		data <= 32'd0;
	end
	else
	begin
		case (S)
			WAIT_WREN:
			begin
				o_write_ready <= 1'b1; //indicates waiting for next input
				data <= i_data;  //save current data (avoids strange behavior if input is changed asynchronously)
			end
			COL_CHECK:
			begin
				o_write_ready <= 1'b0; //indicates operation
				col_num <= (NS == ROW_INC)?(4'd0):(col_num); //resetting column number when exiting for loop
			end
			WRITE:
			begin
				ascii_wren <= 1'b1; //write to ASCII memory
			end
			COL_INC:
			begin
				ascii_wren <= 1'b0;
				col_num <= col_num + 4'd1; //increment column number
			end
			ROW_INC:
			begin
				row_num <= row_num + 8'd1; //increment row number
			end
			default:; //do nothing
		endcase
	end
end

//decoding wires
wire is_negative;
wire [31:0]two_comp;
wire [3:0]current_char_val;

assign ascii_addr = (13'd80 * (row_num % 8'd60)) + col_num + ((row_num / 60) * 8'd10); //decode address
assign is_negative = data[31]; //check sign
assign two_comp = (is_negative == 1'b1)?((~data) + 1'b1):data; //convert from two's complement
assign current_char_val = two_comp >> 4*(4'd8-col_num);//find value of current character from two's comp and column number

//hex to ascii decoding table
always @ (*)
begin
	if (col_num == 4'b0)
	begin
		ascii_code = (is_negative)?(8'h2D):(8'h2B);
	end
	else
	begin
		case (current_char_val)
			4'h0: ascii_code = 8'b00110000;
			4'h1: ascii_code = 8'b00110001;
			4'h2: ascii_code = 8'b00110010;
			4'h3: ascii_code = 8'b00110011;
			4'h4: ascii_code = 8'b00110100;
			4'h5: ascii_code = 8'b00110101;
			4'h6: ascii_code = 8'b00110110;
			4'h7: ascii_code = 8'b00110111;
			4'h8: ascii_code = 8'b00111000;
			4'h9: ascii_code = 8'b00111001;
			4'hA: ascii_code = 8'b01000001;
			4'hB: ascii_code = 8'b01000010;
			4'hC: ascii_code = 8'b01000011;
			4'hD: ascii_code = 8'b01000100;
			4'hE: ascii_code = 8'b01000101;
			4'hF: ascii_code = 8'b01000110;
			default: ascii_code = 8'h24; //display $ if decode error occurs
		endcase
	end
end



assign ascii_data = {ascii_code, 24'hFFFFFF}; //assign color of text to be white

//ASCII VGA Controller instantiation
ascii_master_controller controller (

	.clk(i_clk),
	.rst(i_rst),
	
	.ascii_write_en(ascii_wren), //enables writing
	.ascii_input(ascii_data), 
	.ascii_write_address(ascii_addr), //test address is 20
	
	//VGA signal passthrough
	.vga_blank(vga_blank),
	.vga_b(vga_b),
	.vga_r(vga_r),
	.vga_g(vga_g),
	.vga_clk(vga_clk),
	.vga_hs(vga_hs),
	.vga_vs(vga_vs),
	.vga_sync(vga_sync)

	// ---------- DEBUG ---------- //
	//.SW(SW[9:0]),
	//.KEY(KEY[3:0]),
	//.LEDR(LEDR[9:0])
	// ---------- END DEBUG ---------- //
);
// ---------- END CODE ---------- //


// ---------- DEBUG ---------- //
/*
assign LEDR[2:0] = S;
assign LEDR[9:6] = col_num;
*/
// ---------- END DEBUG ---------- //

endmodule