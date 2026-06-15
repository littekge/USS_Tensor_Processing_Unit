/* 
 * File: TPU.v
 * Author: Gabe Litteken
 * Date: 2026-06-15
 * 
 * Top-level module for the Functional TPU. Instantiates the Programmer,
 * Program_Memory, Weight_Memory, TPU_0x1_Buffer, Feeder, and Controller.
 * Defines the trst signal and implements the memory MUX. Controller clear
 * is combined with inverted trst for the vector buffer sclr. Vector_Processor,
 * ALU, Activator, and Systolic_Array connections are stubbed for steps 4-7.
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
// Feeder controls Program_Memory when program=HIGH.
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

// Memory read outputs
wire [127:0] pm_q;
wire [7:0]   wm_q_a, wm_q_b;
wire [7:0]   buf_q_a, buf_q_b;

// Feeder program memory and controller outputs
wire [9:0]   feeder_pm_address;
wire         feeder_pm_wren;
wire [127:0] feeder_instruction;
wire         feeder_controller_start;

// Controller outputs — Feeder handshake
wire         ctrl_controller_idle;
wire         ctrl_clear;

// Controller outputs — Vector_Processor_A control
// (connected to Vector_Processor_A in build step 4)
wire         ctrl_vector_start_a;
wire [2:0]   ctrl_vect_source_a;
wire [2:0]   ctrl_vect_dest_a;
wire [15:0]  ctrl_mem_source_address_a;
wire [15:0]  ctrl_mem_dest_address_a;
wire [15:0]  ctrl_length_a;
wire [3:0]   ctrl_dim0_a;
wire [3:0]   ctrl_dim1_a;

// Controller outputs — Vector_Processor_B control
// (connected to Vector_Processor_B in build step 4)
wire         ctrl_vector_start_b;
wire [2:0]   ctrl_vect_source_b;
wire [2:0]   ctrl_vect_dest_b;
wire [15:0]  ctrl_mem_source_address_b;
wire [15:0]  ctrl_length_b;
wire [3:0]   ctrl_dim0_b;
wire [3:0]   ctrl_dim1_b;

// Controller outputs — Activator and ALU function selects
// (connected in build steps 5 and 6)
wire [2:0]   ctrl_activator_funct;
wire [2:0]   ctrl_alu_funct;

// Controller output — Systolic_Array start pulse
// (connected in build step 7)
wire         ctrl_systolic_array_start;

// Vector buffer sclr: asserted when trst is LOW or controller clear is HIGH
wire vb_sclr;
assign vb_sclr = ctrl_clear | ~trst;

assign pm_data    = (program == 1'b0) ? prog_pm_data    : 128'd0;
assign pm_address = (program == 1'b0) ? prog_pm_address : feeder_pm_address;
assign pm_wren    = (program == 1'b0) ? prog_pm_wren    : 1'b0;

assign wm_data_a    = (program == 1'b0) ? prog_wm_data_a    : 8'd0;
assign wm_address_a = (program == 1'b0) ? prog_wm_address_a : 16'd0;
assign wm_wren_a    = (program == 1'b0) ? prog_wm_wren_a    : 1'b0;

assign buf_data_a    = (program == 1'b0) ? prog_buf_data_a    : 8'd0;
assign buf_address_a = (program == 1'b0) ? prog_buf_address_a : 16'd0;
assign buf_wren_a    = (program == 1'b0) ? prog_buf_wren_a    : 1'b0;

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

Feeder feeder (
    .i_clk              (i_clk),
    .i_trst             (trst),
    .i_pm_q             (pm_q),
    .o_pm_address       (feeder_pm_address),
    .o_pm_wren          (feeder_pm_wren),
    .i_controller_idle  (ctrl_controller_idle),
    .o_instruction      (feeder_instruction),
    .o_controller_start (feeder_controller_start)
);

Controller ctrl (
    .i_clk                  (i_clk),
    .i_trst                 (trst),
    .i_controller_start     (feeder_controller_start),
    .i_instruction          (feeder_instruction),
    .i_vector_idle_a        (1'b1),   // stub: connected to Vector_Processor_A in step 4
    .i_vector_idle_b        (1'b1),   // stub: connected to Vector_Processor_B in step 4
    .i_systolic_array_idle  (1'b1),   // stub: connected to Systolic_Array in step 7
    .o_controller_idle      (ctrl_controller_idle),
    .o_clear                (ctrl_clear),
    .o_vector_start_a       (ctrl_vector_start_a),
    .o_vect_source_a        (ctrl_vect_source_a),
    .o_vect_dest_a          (ctrl_vect_dest_a),
    .o_mem_source_address_a (ctrl_mem_source_address_a),
    .o_mem_dest_address_a   (ctrl_mem_dest_address_a),
    .o_length_a             (ctrl_length_a),
    .o_dim0_a               (ctrl_dim0_a),
    .o_dim1_a               (ctrl_dim1_a),
    .o_vector_start_b       (ctrl_vector_start_b),
    .o_vect_source_b        (ctrl_vect_source_b),
    .o_vect_dest_b          (ctrl_vect_dest_b),
    .o_mem_source_address_b (ctrl_mem_source_address_b),
    .o_length_b             (ctrl_length_b),
    .o_dim0_b               (ctrl_dim0_b),
    .o_dim1_b               (ctrl_dim1_b),
    .o_activator_funct      (ctrl_activator_funct),
    .o_alu_funct            (ctrl_alu_funct),
    .o_systolic_array_start (ctrl_systolic_array_start)
);

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
