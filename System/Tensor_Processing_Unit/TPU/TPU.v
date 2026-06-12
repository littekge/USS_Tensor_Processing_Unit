/* 
 * File: TPU.v
 * Author: Gabe Litteken
 * Date: 2026-06-12
 * 
 * Top-level module for the Functional TPU. Instantiates the Programmer,
 * Program_Memory, Weight_Memory, and TPU_0x1_Buffer. Defines the trst
 * signal and implements a MUX that routes memory control signals between
 * the Programmer (when program is LOW) and future Feeder/Vector_Processor
 * modules (when program is HIGH).
 */
module TPU (

    input       i_clk,
    input       i_rst,

    // ---------- PARAMETERS ---------- //
    // ---------- END PARAMETERS ---------- //

    // SPI signals (passed through to Programmer)
    input       i_SPI_Clk,
    output wire o_SPI_MISO,
    input       i_SPI_MOSI,
    input       i_SPI_SS

    // ---------- DEBUG ---------- //
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// Programmer memory write outputs
wire [127:0] prog_pm_data;
wire [9:0]   prog_pm_address;
wire         prog_pm_wren;
wire [7:0]   prog_wm_data_a;
wire [15:0]  prog_wm_address_a;
wire         prog_wm_wren_a;
wire [7:0]   prog_buf_data_a;
wire [15:0]  prog_buf_address_a;
wire         prog_buf_wren_a;

// program: LOW while Programmer is actively programming the TPU
wire program;

// trst: LOW when global reset (rst) OR program signal is LOW.
// Prevents TPU internals from modifying memory during programming.
wire trst;
assign trst = i_rst & program;

// MUX-selected inputs to memories.
// Programmer controls all ports when program=LOW.
// Feeder controls Program_Memory when program=HIGH (stub: connected in step 2).
// Vector_Processor A controls Weight_Memory and 0x1 Buffer port A when
// program=HIGH (stub: connected in step 4).
wire [127:0] pm_data;
wire [9:0]   pm_address;
wire         pm_wren;
wire [7:0]   wm_data_a;
wire [15:0]  wm_address_a;
wire         wm_wren_a;
wire [7:0]   buf_data_a;
wire [15:0]  buf_address_a;
wire         buf_wren_a;

assign pm_data    = (program == 1'b0) ? prog_pm_data    : 128'd0;
assign pm_address = (program == 1'b0) ? prog_pm_address : 10'd0;
assign pm_wren    = (program == 1'b0) ? prog_pm_wren    : 1'b0;

assign wm_data_a    = (program == 1'b0) ? prog_wm_data_a    : 8'd0;
assign wm_address_a = (program == 1'b0) ? prog_wm_address_a : 16'd0;
assign wm_wren_a    = (program == 1'b0) ? prog_wm_wren_a    : 1'b0;

assign buf_data_a    = (program == 1'b0) ? prog_buf_data_a    : 8'd0;
assign buf_address_a = (program == 1'b0) ? prog_buf_address_a : 16'd0;
assign buf_wren_a    = (program == 1'b0) ? prog_buf_wren_a    : 1'b0;

// Memory read outputs (used by Feeder and Vector_Processor in later build steps)
wire [127:0] pm_q;
wire [7:0]   wm_q_a, wm_q_b;
wire [7:0]   buf_q_a, buf_q_b;

Programmer prog (
    .i_clk          (i_clk),
    .i_rst          (i_rst),
    .o_pm_data      (prog_pm_data),
    .o_pm_address   (prog_pm_address),
    .o_pm_wren      (prog_pm_wren),
    .o_wm_data_a    (prog_wm_data_a),
    .o_wm_address_a (prog_wm_address_a),
    .o_wm_wren_a    (prog_wm_wren_a),
    .o_buf_data_a   (prog_buf_data_a),
    .o_buf_address_a(prog_buf_address_a),
    .o_buf_wren_a   (prog_buf_wren_a),
    .o_program      (program),
    .i_SPI_Clk      (i_SPI_Clk),
    .o_SPI_MISO     (o_SPI_MISO),
    .i_SPI_MOSI     (i_SPI_MOSI),
    .i_SPI_SS       (i_SPI_SS)
);

Program_Memory pm (
    .clock  (i_clk),
    .data   (pm_data),
    .address(pm_address),
    .wren   (pm_wren),
    .q      (pm_q)
);

Weight_Memory wm (
    .clock    (i_clk),
    .data_a   (wm_data_a),
    .address_a(wm_address_a),
    .wren_a   (wm_wren_a),
    .data_b   (8'd0),    // connected to Vector_Processor B in build step 4
    .address_b(16'd0),
    .wren_b   (1'b0),
    .q_a      (wm_q_a),
    .q_b      (wm_q_b)
);

TPU_0x1_Buffer buff (
    .clock    (i_clk),
    .data_a   (buf_data_a),
    .address_a(buf_address_a),
    .wren_a   (buf_wren_a),
    .data_b   (8'd0),    // connected to Vector_Processor B in build step 4
    .address_b(16'd0),
    .wren_b   (1'b0),
    .q_a      (buf_q_a),
    .q_b      (buf_q_b)
);

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
