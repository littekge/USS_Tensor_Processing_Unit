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

//ensure that Quartus knows that these GPIO pins are inputs
assign GPIO_0[7] = 1'bz;
assign GPIO_0[3] = 1'bz;
assign GPIO_0[1] = 1'bz;

// ---------- CODE ---------- //

// Assign unused hardware outputs to safe defaults
assign VGA_BLANK_N = 1'b1;
assign VGA_CLK     = 1'b0;
assign VGA_SYNC_N  = 1'b0;
assign VGA_HS      = 1'b0;
assign VGA_VS      = 1'b0;
assign VGA_R       = 8'd0;
assign VGA_G       = 8'd0;
assign VGA_B       = 8'd0;
assign LEDR        = 10'd0;

// SPI signals routed through GPIO_0 (same pins as prior SPI test)
//   GPIO_0[7] = SPI_Clk (input)
//   GPIO_0[5] = SPI_MISO (output, tri-stated when inactive)
//   GPIO_0[3] = SPI_MOSI (input)
//   GPIO_0[1] = SPI_SS (input, active LOW)
TPU tpu (
	.i_clk    (clk),
	.i_rst    (rst),
	.i_SPI_Clk(GPIO_0[7]),
	.o_SPI_MISO(GPIO_0[5]),
	.i_SPI_MOSI(GPIO_0[3]),
	.i_SPI_SS (GPIO_0[1])
);

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule