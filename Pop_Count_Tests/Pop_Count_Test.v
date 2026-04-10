module Pop_Count_Test(
  
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
	input 		          		CLOCK_50,

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
	output		     [6:0]		HEX4,
	output		     [6:0]		HEX5,

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
	output		          		VGA_VS

	//////////// GPIO_0, GPIO_0 connect to GPIO Default //////////
	// inout 		    [35:0]		GPIO_0,

	//////////// GPIO_1, GPIO_1 connect to GPIO Default //////////
	// inout 		    [35:0]		GPIO_1
);

	localparam [1:0] S_IDLE  = 2'b00;
	localparam [1:0] S_LOAD  = 2'b01;
	localparam [1:0] S_COUNT = 2'b10;
	localparam [1:0] S_DONE  = 2'b11;

	reg [1:0] state;
	reg [1:0] next_state;
	reg [7:0] working_value;
	reg [2:0] bit_index;
	reg [3:0] count_reg;
	reg [3:0] result_reg;

	always @(posedge CLOCK_50 or negedge KEY[0]) begin
		if (!KEY[0]) begin
			state <= S_IDLE;
		end else begin
			state <= next_state;
		end
	end

	always @(*) begin
		next_state = state;

		case (state)
			S_IDLE: begin
				next_state = S_LOAD;
			end

			S_LOAD: begin
				next_state = S_COUNT;
			end

			S_COUNT: begin
				if (bit_index == 3'd7) begin
					next_state = S_DONE;
				end else begin
					next_state = S_COUNT;
				end
			end

			S_DONE: begin
				next_state = S_LOAD;
			end

			default: begin
				next_state = S_IDLE;
			end
		endcase
	end

	always @(posedge CLOCK_50 or negedge KEY[0]) begin
		if (!KEY[0]) begin
			working_value <= 8'b0;
			bit_index <= 3'b0;
			count_reg <= 4'b0;
			result_reg <= 4'b0;
		end else begin
			case (state)
				S_IDLE: begin
					count_reg <= 4'b0;
					bit_index <= 3'b0;
					working_value <= 8'b0;
				end

				S_LOAD: begin
					working_value <= SW[7:0];
					count_reg <= 4'b0;
					bit_index <= 3'b0;
				end

				S_COUNT: begin
					count_reg <= count_reg + working_value[0];
					working_value <= {1'b0, working_value[7:1]};
					bit_index <= bit_index + 3'b001;
				end

				S_DONE: begin
					result_reg <= count_reg;
				end
			endcase
		end
	end

	seven_segment_decoder u_hex0_decoder (
		.value(result_reg),
		.segments(HEX0)
	);
	assign HEX1 = 7'b1111111;
	assign HEX2 = 7'b1111111;
	assign HEX3 = 7'b1111111;
	assign HEX4 = 7'b1111111;
	assign HEX5 = 7'b1111111;

	assign LEDR = {result_reg[1:0], SW[7:0]};

	assign VGA_BLANK_N = 1'b1;
	assign VGA_B = 8'b00000000;
	assign VGA_CLK = CLOCK_50;
	assign VGA_G = 8'b00000000;
	assign VGA_HS = 1'b1;
	assign VGA_R = 8'b00000000;
	assign VGA_SYNC_N = 1'b1;
	assign VGA_VS = 1'b1;




endmodule

module seven_segment_decoder(
	input [3:0] value,
	output reg [6:0] segments
);

	always @(*) begin
		case (value)
			4'd0: segments = 7'b1000000;
			4'd1: segments = 7'b1111001;
			4'd2: segments = 7'b0100100;
			4'd3: segments = 7'b0110000;
			4'd4: segments = 7'b0011001;
			4'd5: segments = 7'b0010010;
			4'd6: segments = 7'b0000010;
			4'd7: segments = 7'b1111000;
			4'd8: segments = 7'b0000000;
			4'd9: segments = 7'b0010000;
			default: segments = 7'b1111111;
		endcase
	end

endmodule