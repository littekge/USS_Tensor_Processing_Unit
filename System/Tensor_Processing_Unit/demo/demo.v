/*
 * File: demo.v
 * Author: Gabe Litteken
 * Date: 7/22/2026
 *
 * Presentation top-level for the Functional TPU. It carries the full
 * peripheral wiring of Tensor_Processing_Unit.v (SPI-over-GPIO, mode select,
 * device-mode input, HEX, LEDR) but replaces the raw-hex debug.v VGA dump with
 * a LeNet-5 result view: the 10 class outputs are collected, negatives are
 * treated as zero, and the classes are shown most- to least-likely as
 * normalized percentages (one decimal place) on the VGA screen.
 */
module demo (
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

// ---------- PARAMETERS ---------- //
// Result pipeline states. START has no execution logic (per convention).
parameter
	START      = 4'd0,  // idle after reset
	ARM        = 4'd1,  // wait for end-of-inference strobe
	COLLECT    = 4'd2,  // latch the CLASS_COUNT streamed outputs
	SORT_INIT  = 4'd3,  // seed the ranking order with 0..CLASS_COUNT-1
	SORT_SETUP = 4'd4,  // begin one selection-sort pass
	SORT_SCAN  = 4'd5,  // scan the remaining tail for the largest class
	SORT_SWAP  = 4'd6,  // place the largest class at the current rank
	DIV_SETUP  = 4'd7,  // begin a percentage division for one class
	DIV_STEP   = 4'd8,  // repeated-subtraction division step
	CLEAR      = 4'd9,  // blank the whole character buffer
	WRITE      = 4'd10, // render header + ranked lines
	HOLD       = 4'd11; // display held; re-arm on the next inference

// LeNet-5 emits ten class scores; the Programmer streams exactly this many in
// external-output mode (EXT_OUTPUT_COUNT), in class order 0..9.
parameter integer CLASS_COUNT = 10;

// Character grid geometry of the ASCII VGA controller (60 rows x 80 columns).
parameter integer GRID_COLS   = 80;
parameter integer GRID_CELLS  = 4800;   // 60 * 80
parameter integer CONTENT_TOP = 2;      // first row of ranked class lines
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
wire clk, rst;
assign clk = CLOCK_50;
assign rst = KEY[0];

// Tell Quartus these GPIO pins are inputs (SPI clock/MOSI/SS).
assign GPIO_0[7] = 1'bz;
assign GPIO_0[3] = 1'bz;
assign GPIO_0[1] = 1'bz;

// LOW = device mode (manual SW/HEX), HIGH = external mode (streamed input).
wire mode_select;
assign mode_select = SW[9];

wire [7:0] input_data;
assign input_data = (mode_select == 1'b0) ? SW[7:0] : 8'd0;

wire input_data_valid;
assign input_data_valid = (mode_select == 1'b0) ? (~KEY[1]) : 1'd0;

// HEX shows the raw streamed value in device mode, blank in external mode.
wire [6:0] hex_out[0:3];
assign HEX0 = (mode_select == 1'b0) ? hex_out[0] : 7'b1111111;
assign HEX1 = (mode_select == 1'b0) ? hex_out[1] : 7'b1111111;
assign HEX2 = (mode_select == 1'b0) ? hex_out[2] : 7'b1111111;
assign HEX3 = (mode_select == 1'b0) ? hex_out[3] : 7'b1111111;

wire       output_valid;
wire [7:0] output_val;
wire       end_reached;

// The result view consumes outputs as fast as the TPU offers them; there is no
// downstream backpressure to model, so the consumer is always ready.
wire device_ready;
assign device_ready = 1'b1;

three_decimal_vals_w_neg seg (
	.val(output_val),
	.seg7_neg_sign(hex_out[3]),
	.seg7_dig0(hex_out[0]),
	.seg7_dig1(hex_out[1]),
	.seg7_dig2(hex_out[2])
);

// SPI signals routed through GPIO_0 (same pins as the standard top-level):
//   GPIO_0[7] = SPI_Clk, GPIO_0[5] = SPI_MISO, GPIO_0[3] = SPI_MOSI,
//   GPIO_0[1] = SPI_SS (active LOW).
TPU tpu (
	.i_clk    (clk),
	.i_rst    (rst),
	.i_SPI_Clk(GPIO_0[7]),
	.o_SPI_MISO(GPIO_0[5]),
	.i_SPI_MOSI(GPIO_0[3]),
	.i_SPI_SS (GPIO_0[1]),

	.i_mode_select(mode_select),
	.i_input_data_valid(input_data_valid),
	.i_device_ready(device_ready),
	.i_input_data(input_data),
	.o_output_data(output_val),
	.o_output_data_valid(output_valid),
	.o_end_reached(end_reached),

	.o_error_code(error_code)
);

// ----- Result pipeline: collect, rank, and render as percentages ----- //
integer k; //initialization/reduction loop variable

reg [7:0]  results [0:CLASS_COUNT-1]; //raw signed class scores, indexed by digit
reg [3:0]  order   [0:CLASS_COUNT-1]; //class indices, ranked most- to least-likely
reg [10:0] tenths  [0:CLASS_COUNT-1]; //percentage*10 per rank (0..1000 = 0.0..100.0%)

reg [3:0]  val_cnt;   //number of outputs collected this inference
reg [3:0]  sort_i;    //selection-sort current rank
reg [3:0]  sort_j;    //selection-sort scan index
reg [3:0]  sort_best; //index of the largest class found so far in the scan
reg [3:0]  div_li;    //rank currently being converted to a percentage
reg [16:0] num;       //dividend remainder for repeated-subtraction division
reg [10:0] q;         //quotient accumulator (percentage*10)

reg [12:0] clear_addr; //buffer-clear sweep address
reg [7:0]  wr_row;     //render row
reg [7:0]  wr_col;     //render column

// Negatives are treated as zero for both ranking and normalization.
reg [7:0]  clamped [0:CLASS_COUNT-1];
always @(*)
begin
	for (k = 0; k < CLASS_COUNT; k = k + 1)
		clamped[k] = results[k][7] ? 8'd0 : results[k];
end

// Normalization denominator: sum of the clamped scores.
reg [10:0] sum_clamped;
always @(*)
begin
	sum_clamped = 11'd0;
	for (k = 0; k < CLASS_COUNT; k = k + 1)
		sum_clamped = sum_clamped + {3'd0, clamped[k]};
end

// State machine variables
reg [3:0] S, NS;

// State machine driver
always @ (posedge clk or negedge rst)
begin
	if (rst == 1'b0)
	begin
		S <= START;
	end
	else
	begin
		S <= NS;
	end
end

// Determining next state
always @ (*)
begin
	case (S)
		START:      NS = ARM;
		// end_reached strobes once when the inference finishes, just before the
		// outputs stream out; it is the natural per-inference arm signal.
		ARM:        NS = (end_reached == 1'b1) ? COLLECT : ARM;
		COLLECT:    NS = (val_cnt == CLASS_COUNT) ? SORT_INIT : COLLECT;
		SORT_INIT:  NS = SORT_SETUP;
		SORT_SETUP: NS = (sort_i >= (CLASS_COUNT - 1)) ? DIV_SETUP : SORT_SCAN;
		SORT_SCAN:  NS = (sort_j == CLASS_COUNT) ? SORT_SWAP : SORT_SCAN;
		SORT_SWAP:  NS = SORT_SETUP;
		DIV_SETUP:  NS = (div_li == CLASS_COUNT) ? CLEAR :
		                 (sum_clamped == 11'd0)  ? DIV_SETUP : DIV_STEP;
		DIV_STEP:   NS = (num >= {6'd0, sum_clamped}) ? DIV_STEP : DIV_SETUP;
		CLEAR:      NS = (clear_addr == (GRID_CELLS - 1)) ? WRITE : CLEAR;
		WRITE:      NS = (wr_row == 8'd11 && wr_col == 8'd79) ? HOLD : WRITE;
		HOLD:       NS = (end_reached == 1'b1) ? COLLECT : HOLD;
		default:    NS = START;
	endcase
end

// Execution logic
always @ (posedge clk or negedge rst)
begin
	if (rst == 1'b0)
	begin
		val_cnt    <= 4'd0;
		sort_i     <= 4'd0;
		sort_j     <= 4'd0;
		sort_best  <= 4'd0;
		div_li     <= 4'd0;
		num        <= 17'd0;
		q          <= 11'd0;
		clear_addr <= 13'd0;
		wr_row     <= 8'd0;
		wr_col     <= 8'd0;
		for (k = 0; k < CLASS_COUNT; k = k + 1)
		begin
			results[k] <= 8'd0;
			order[k]   <= 4'd0;
			tenths[k]  <= 11'd0;
		end
	end
	else
	begin
		case (S)
			ARM:
			begin
				if (end_reached == 1'b1)
					val_cnt <= 4'd0;
			end
			COLLECT:
			begin
				if (output_valid == 1'b1)
				begin
					results[val_cnt] <= output_val;
					val_cnt          <= val_cnt + 4'd1;
				end
			end
			SORT_INIT:
			begin
				for (k = 0; k < CLASS_COUNT; k = k + 1)
					order[k] <= k[3:0];
				sort_i <= 4'd0;
				div_li <= 4'd0;
			end
			SORT_SETUP:
			begin
				sort_best <= sort_i;
				sort_j    <= sort_i + 4'd1;
			end
			SORT_SCAN:
			begin
				if (sort_j < CLASS_COUNT)
				begin
					// Strict greater-than keeps the lower digit first on ties.
					if (clamped[order[sort_j]] > clamped[order[sort_best]])
						sort_best <= sort_j;
					sort_j <= sort_j + 4'd1;
				end
			end
			SORT_SWAP:
			begin
				order[sort_i]    <= order[sort_best];
				order[sort_best] <= order[sort_i];
				sort_i           <= sort_i + 4'd1;
			end
			DIV_SETUP:
			begin
				if (div_li == CLASS_COUNT)
				begin
					clear_addr <= 13'd0;
				end
				else if (sum_clamped == 11'd0)
				begin
					// All scores <= 0: every class normalizes to 0.0%.
					tenths[div_li] <= 11'd0;
					div_li         <= div_li + 4'd1;
				end
				else
				begin
					// Repeated subtraction (not a single-cycle divide) keeps the
					// variable-divisor normalization inside timing; the VGA buffer
					// only needs the result eventually, so the extra cycles are free.
					num <= clamped[order[div_li]] * 17'd1000;
					q   <= 11'd0;
				end
			end
			DIV_STEP:
			begin
				if (num >= {6'd0, sum_clamped})
				begin
					num <= num - {6'd0, sum_clamped};
					q   <= q + 11'd1;
				end
				else
				begin
					tenths[div_li] <= q;
					div_li         <= div_li + 4'd1;
				end
			end
			CLEAR:
			begin
				clear_addr <= clear_addr + 13'd1;
				if (clear_addr == (GRID_CELLS - 1))
				begin
					wr_row <= 8'd0;
					wr_col <= 8'd0;
				end
			end
			WRITE:
			begin
				if (wr_col == 8'd79)
				begin
					wr_col <= 8'd0;
					wr_row <= wr_row + 8'd1;
				end
				else
				begin
					wr_col <= wr_col + 8'd1;
				end
			end
			HOLD:
			begin
				if (end_reached == 1'b1)
					val_cnt <= 4'd0;
			end
			default:; //START and transient states have no execution logic
		endcase
	end
end

// ----- Character generation for the current render cell ----- //
// Percentage of the class on the row being written (rows CONTENT_TOP..11).
wire [3:0]  line_rank    = wr_row - CONTENT_TOP;
wire [3:0]  line_digit   = order[line_rank];
wire [10:0] line_tenths  = tenths[line_rank];
wire [10:0] int_part     = line_tenths / 11'd10;   // whole percent, 0..100
wire [3:0]  frac_digit   = line_tenths % 11'd10;   // tenths of a percent
wire [3:0]  hund_digit   = (int_part / 11'd100) % 11'd10;
wire [3:0]  tens_digit   = (int_part / 11'd10)  % 11'd10;
wire [3:0]  ones_digit   = int_part % 11'd10;

reg [7:0] cell_char;
always @(*)
begin
	cell_char = 8'h20; //space
	if (wr_row == 8'd0)
	begin
		// Header text: "LeNet-5 Prediction".
		case (wr_col)
			8'd0:  cell_char = 8'h4C; // L
			8'd1:  cell_char = 8'h65; // e
			8'd2:  cell_char = 8'h4E; // N
			8'd3:  cell_char = 8'h65; // e
			8'd4:  cell_char = 8'h74; // t
			8'd5:  cell_char = 8'h2D; // -
			8'd6:  cell_char = 8'h35; // 5
			8'd8:  cell_char = 8'h50; // P
			8'd9:  cell_char = 8'h72; // r
			8'd10: cell_char = 8'h65; // e
			8'd11: cell_char = 8'h64; // d
			8'd12: cell_char = 8'h69; // i
			8'd13: cell_char = 8'h63; // c
			8'd14: cell_char = 8'h74; // t
			8'd15: cell_char = 8'h69; // i
			8'd16: cell_char = 8'h6F; // o
			8'd17: cell_char = 8'h6E; // n
			default: cell_char = 8'h20;
		endcase
	end
	else if (wr_row >= CONTENT_TOP && wr_row <= 8'd11)
	begin
		// "  D    HTO.F%" with leading zeros of the whole part blanked.
		case (wr_col)
			8'd2:  cell_char = 8'h30 + {4'd0, line_digit};
			8'd6:  cell_char = (int_part >= 11'd100) ? (8'h30 + {4'd0, hund_digit}) : 8'h20;
			8'd7:  cell_char = (int_part >= 11'd10)  ? (8'h30 + {4'd0, tens_digit}) : 8'h20;
			8'd8:  cell_char = 8'h30 + {4'd0, ones_digit};
			8'd9:  cell_char = 8'h2E; // .
			8'd10: cell_char = 8'h30 + {4'd0, frac_digit};
			8'd11: cell_char = 8'h25; // %
			default: cell_char = 8'h20;
		endcase
	end
end

// ----- VGA write port ----- //
reg        ascii_wren;
reg [12:0] ascii_addr;
always @(*)
begin
	ascii_wren = 1'b0;
	ascii_addr = 13'd0;
	case (S)
		CLEAR:
		begin
			ascii_wren = 1'b1;
			ascii_addr = clear_addr;
		end
		WRITE:
		begin
			ascii_wren = 1'b1;
			ascii_addr = (wr_row * GRID_COLS) + wr_col;
		end
		default:;
	endcase
end

// White text; blank (all-zero) fill while clearing.
wire [31:0] ascii_data;
assign ascii_data = (S == CLEAR) ? 32'd0 : {cell_char, 24'hFFFFFF};

ascii_master_controller vga (
	.clk(clk),
	.rst(rst),

	.ascii_write_en(ascii_wren),
	.ascii_input(ascii_data),
	.ascii_write_address(ascii_addr),

	.vga_blank(VGA_BLANK_N),
	.vga_b(VGA_B),
	.vga_r(VGA_R),
	.vga_g(VGA_G),
	.vga_clk(VGA_CLK),
	.vga_hs(VGA_HS),
	.vga_vs(VGA_VS),
	.vga_sync(VGA_SYNC_N)
);
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
wire [2:0] error_code;

reg [9:0] LEDS;
assign LEDR[9:0] = LEDS[9:0];

always @ (*)
begin
	LEDS[9] = SW[9];
	if (error_code == 3'd0)
	begin
		LEDS[8]   = 1'd0;
		LEDS[7:0] = SW[7:0];
	end
	else
	begin
		LEDS[8]   = 1'd1;
		LEDS[7:0] = {5'd0, error_code};
	end
end
// ---------- END DEBUG ---------- //
endmodule
