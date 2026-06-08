/* 
 * File: Tensor_Processing_Unit.v
 * Author: Gabe Litteken 
 * Date: 5/19/2026
 * 
 * This is the top level module for this project. It is designed to implement
 * a tensor processing unit based on Google's design and interface it with
 * relevant peripherals.
 */
module Tensor_Processing_Unit (
	//////////// ADC //////////
	// output		          		ADC_CONVST,
	// output		          		ADC_DIN,
	// input 		          		ADC_DOUT,
	// output		          		ADC_SCLK,

	//////////// Audio //////////
	// input 		          		AUD_ADCDAT,
	// inout 		          		AUD_ADCLRCK,
	// inout 		          		AUD_BCLK,
	// output		          		AUD_DACDAT,
	// inout 		          		AUD_DACLRCK,
	// output		          		AUD_XCK,

	//////////// CLOCK //////////
	// input 		          		CLOCK2_50,
	// input 		          		CLOCK3_50,
	// input 		          		CLOCK4_50,
	input 		          			CLOCK_50,

	//////////// SDRAM //////////
	// output		    [12:0]		DRAM_ADDR,
	// output		     [1:0]		DRAM_BA,
	// output		          		DRAM_CAS_N,
	// output		          		DRAM_CKE,
	// output		          		DRAM_CLK,
	// output		          		DRAM_CS_N,
	// inout 		    [15:0]		DRAM_DQ,
	// output		          		DRAM_LDQM,
	// output		          		DRAM_RAS_N,
	// output		          		DRAM_UDQM,
	// output		          		DRAM_WE_N,

	//////////// I2C for Audio and Video-In //////////
	// output		          		FPGA_I2C_SCLK,
	// inout 		          		FPGA_I2C_SDAT,

	//////////// SEG7 //////////
	//output		     [6:0]		HEX0,
	//output		     [6:0]		HEX1,
	//output		     [6:0]		HEX2,
	//output		     [6:0]		HEX3,
	//output		     [6:0]		HEX4,
	//output		     [6:0]		HEX5,

	//////////// IR //////////
	// input 		          		IRDA_RXD,
	// output		          		IRDA_TXD,

	//////////// KEY //////////
	input 		     [3:0]		KEY,

	//////////// LED //////////
	output		     [9:0]		LEDR,

	//////////// PS2 //////////
	// inout 		          		PS2_CLK,
	// inout 		          		PS2_CLK2,
	// inout 		          		PS2_DAT,
	// inout 		          		PS2_DAT2,

	//////////// SW //////////
	input 		     [9:0]		SW,

	//////////// Video-In //////////
	// input 		          		TD_CLK27,
	// input 		     [7:0]		TD_DATA,
	// input 		          		TD_HS,
	// output		          		TD_RESET_N,
	// input 		          		TD_VS,

	//////////// VGA //////////
	output		          		VGA_BLANK_N,
	output		     [7:0]		VGA_B,
	output		          		VGA_CLK,
	output		     [7:0]		VGA_G,
	output		          		VGA_HS,
	output		     [7:0]		VGA_R,
	output		          		VGA_SYNC_N,
	output		          		VGA_VS,

	//////////// GPIO_0, GPIO_0 connect to GPIO Default //////////
	inout 		    [35:0]		GPIO_0

	//////////// GPIO_1, GPIO_1 connect to GPIO Default //////////
	// inout 		    [35:0]		GPIO_1


);
wire clk, rst;
assign clk = CLOCK_50;
assign rst = KEY[0];




// ---------- DEBUG ---------- //

//code to test SPI connection
wire readrq;
assign readrq = ~KEY[3];

SPI_Interface SPI_int(
	//clock and reset
   .clk(clk),
   .rst(rst),
	
	//buffer interfacing signals
	.i_rdreq(readrq),
	.o_empty(),
	.o_full(),
	.o_q(),
	.o_usedw(),
	
	//SPI Signals
	.i_SPI_Clk(GPIO_0[7]),
	.o_SPI_MISO(GPIO_0[5]),
	.i_SPI_MOSI(GPIO_0[3]),
	.i_SPI_SS(GPIO_0[1])
);


/*
//code to test ASCII drivers
ascii_master_controller controller (

	.clk(clk),
	.rst(rst),
	
	.ascii_write_en(1'b1), //enables writing
	.ascii_input(32'h37FFFFFF), //test character is a white 7
	.ascii_write_address(13'd20), //test address is 20
	
	.vga_blank(VGA_BLANK_N),
	.vga_b(VGA_B),
	.vga_r(VGA_R),
	.vga_g(VGA_G),
	.vga_clk(VGA_CLK),
	.vga_hs(VGA_HS),
	.vga_vs(VGA_VS),
	.vga_sync(VGA_SYNC_N),

	// ---------- DEBUG ---------- //
	.SW(SW[9:0]),
	.KEY(KEY[3:0]),
	.LEDR(LEDR[9:0])
	// ---------- END DEBUG ---------- //
);
*/


//code to test debug module
/*
parameter
	WAIT_READY = 2'd0,
	WRITE_ON = 2'd1,
	WRITE_OFF = 2'd2,
	INC = 2'd3;

reg [1:0]S, NS;
reg [7:0]count;
reg start_write;

always @ (posedge clk or negedge rst)
begin
	if (rst == 1'b0)
	begin
		S <= WAIT_READY; //sets state to START if reset is triggered
	end
	else
	begin
		S <= NS; //otherwise set S to NS
	end
end

always @ (*)
begin
	case (S)
		WAIT_READY: NS = (write_ready)?(WRITE_ON):(WAIT_READY);
		WRITE_ON: NS = WRITE_OFF;
		WRITE_OFF: NS = INC;
		INC: NS = WAIT_READY;
	endcase
end

always @ (posedge clk or negedge rst)
begin
	if (rst == 1'b0)
	begin
		count <= 8'd0;
		start_write <= 1'd0;
	end
	else
	begin
		case (S)
			WRITE_ON: start_write <= 1'd1;
			WRITE_OFF: start_write <= 1'd0; 
			INC: count <= count + 8'd1;
			default:;
		endcase
	end
end

wire write_ready;

debug debug1 (
	.i_clk(clk),
	.i_rst(rst),
	
	.i_data({24'd0, count}), //signed integer to write to the screen
	.i_write_next(start_write), //pulse high for one clock cycle to write the integer in i_data
	.o_write_ready(write_ready), //high if the module is ready for the next integer
	
	//VGA signal passthrough
	.vga_blank(VGA_BLANK_N),
	.vga_b(VGA_B),
	.vga_r(VGA_R),
	.vga_g(VGA_G),
	.vga_clk(VGA_CLK),
	.vga_hs(VGA_HS),
	.vga_vs(VGA_VS),
	.vga_sync(VGA_SYNC_N),

	// ---------- DEBUG ---------- //
	//.SW(SW[9:0]),
	//.KEY(KEY[3:0]),
	//.LEDR(LEDR[9:0])
	// ---------- END DEBUG ---------- //
);
*/



// ---------- END DEBUG ---------- //
endmodule