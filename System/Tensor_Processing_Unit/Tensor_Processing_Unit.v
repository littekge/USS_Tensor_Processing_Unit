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
	output		     [6:0]		HEX0,
	output		     [6:0]		HEX1,
	output		     [6:0]		HEX2,
	output		     [6:0]		HEX3,
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

// ---------- CODE ---------- //
wire clk, rst;
assign clk = CLOCK_50;
assign rst = KEY[0];

//ensure that Quartus knows that these GPIO pins are inputs
assign GPIO_0[7] = 1'bz;
assign GPIO_0[3] = 1'bz;
assign GPIO_0[1] = 1'bz;

wire display_ready;
wire output_valid;
wire [7:0] output_val;

wire mode_select;
assign mode_select = SW[9];

wire [7:0]input_data;
assign input_data = (mode_select == 1'b0)?(SW[7:0]):(8'd0);

wire input_data_valid;
assign input_data_valid = (mode_select == 1'b0)?(~KEY[1]):(1'd0);

wire [6:0]hex_out[0:3];
assign HEX0 = (mode_select == 1'b0)?(hex_out[0]):(7'b1111111);
assign HEX1 = (mode_select == 1'b0)?(hex_out[1]):(7'b1111111);
assign HEX2 = (mode_select == 1'b0)?(hex_out[2]):(7'b1111111);
assign HEX3 = (mode_select == 1'b0)?(hex_out[3]):(7'b1111111);

wire end_reached;

reg write_next;
always @ (posedge clk or negedge rst)
begin
	if (rst == 1'b0)
	begin
		write_next <= 1'b0;
	end
	else
	begin
		if (output_valid && display_ready) begin
			write_next <= 1'b1;
		end else begin
			write_next <= 1'b0;
		end
	end
end

three_decimal_vals_w_neg seg (
	.val(output_val),
	.seg7_neg_sign(hex_out[3]),
	.seg7_dig0(hex_out[0]),
	.seg7_dig1(hex_out[1]),
	.seg7_dig2(hex_out[2])
);

debug dbg (
   .i_clk(clk),
	.i_rst(rst),
	.i_data(output_val),
	.i_write_next(write_next), 
	.i_clear(end_reached),
	.o_write_ready(display_ready),
	
	//VGA signal passthrough
	.vga_blank(VGA_BLANK_N),
	.vga_b(VGA_B),
	.vga_r(VGA_R),
	.vga_g(VGA_G),
	.vga_clk(VGA_CLK),
	.vga_hs(VGA_HS),
	.vga_vs(VGA_VS),
	.vga_sync(VGA_SYNC_N)
);

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
	.i_SPI_SS (GPIO_0[1]),
	
	.i_mode_select(mode_select),        // LOW = device mode, HIGH = external mode
   .i_input_data_valid(input_data_valid),   // one-cycle pulse: i_input_data valid
   .i_device_ready(display_ready),       // one-cycle pulse: consumer ready for output
   .i_input_data(input_data),         // device-mode input value
   .o_output_data(output_val),        // output value (both modes)
   .o_output_data_valid(output_valid), // one-cycle pulse: o_output_data valid
	.o_end_reached(end_reached),
	
	.o_error_code(error_code)
);

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
wire [2:0] error_code;

reg [9:0] LEDS;
assign LEDR[9:0] = LEDS[9:0];

always @ (*) begin
	LEDS[9] = SW[9];
	if (error_code == 3'd0) begin
		LEDS[8] = 1'd0;
		LEDS[7:0] = SW[7:0];
	end else begin
		LEDS[8] = 1'd1;
		LEDS[7:0] = error_code;
	end
end

// ---------- END DEBUG ---------- //
endmodule