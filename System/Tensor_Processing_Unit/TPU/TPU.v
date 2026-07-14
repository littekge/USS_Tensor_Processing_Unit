/*
 * File: TPU.v
 * Author: Gabe Litteken
 * Date: 2026-06-15
 *
 * Top-level module for the Functional TPU. Instantiates the Programmer,
 * Program_Memory, Data_Memory (unified 0x1 buffer + 2x Mem_Unit), Feeder,
 * Controller, both Vector_Processors, Activator, ALU, Vector_Buffer, and
 * Systolic_Array. Defines the trst signal and implements the memory MUX.
 * Controller clear is combined with inverted trst for the vector buffer sclr.
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
    input       i_SPI_SS,

    // Programmer IO passthrough
    input             i_mode_select,        // LOW = device mode, HIGH = external mode
    input             i_input_data_valid,   // one-cycle pulse: i_input_data valid
    input             i_device_ready,       // HIGH while consumer is ready for the next output value
    input      [7:0]  i_input_data,         // device-mode input value
    output wire [7:0] o_output_data,        // output value (both modes)
    output wire       o_output_data_valid,  // one-cycle pulse: o_output_data valid
	 output wire		 o_end_reached,   // one-cycle pulse when an end instruction is fetched

    // ---------- DEBUG ---------- //
	 output wire [2:0] o_error_code
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// Programmer memory write outputs
wire [127:0] prog_pm_data;
wire [12:0]  prog_pm_address;
wire         prog_pm_wren;
wire [7:0]   prog_dm_data_a;
wire [23:0]  prog_dm_address_a;
wire         prog_dm_wren_a;
wire [9:0]   prog_dm_offset_a;

// program: LOW while the Programmer owns the device memory bus
wire program;

// tpu_rst: LOW while the Programmer is resetting the TPU processing internals
// (during PROGRAM / DEVICE_INPUT / EXTERNAL_INPUT).
wire prog_tpu_rst;

// feeder end_reached strobe (routed to the Programmer to trigger output)
wire feeder_end_reached;
assign o_end_reached = feeder_end_reached;

// trst: LOW when global reset (rst) is LOW OR the Programmer asserts tpu_rst LOW.
// This lets the Programmer reset the TPU internals independently of its memory
// bus access (program), so output meta-states can read memory without resetting
// a finished program.
wire trst;
assign trst = i_rst & prog_tpu_rst;

// MUX-selected inputs to memories.
// Programmer controls all ports when program=LOW.
// Feeder controls Program_Memory when program=HIGH.
// Vector_Processor A controls the Data_Memory port A when program=HIGH.
wire [127:0] pm_data;
wire [12:0]  pm_address;
wire         pm_wren;
wire [7:0]   dm_data_a;
wire [23:0]  dm_address_a;
wire         dm_wren_a;
wire [9:0]   dm_offset_a;

// Memory read outputs
wire [127:0] pm_q;
wire [7:0]   dm_q_a, dm_q_b;

// Data_Memory error flags (observability; the Programmer/Vector_Processor own
// their own error handling for reserved-address writes).
wire dm_write_error_a, dm_write_error_b;
wire dm_range_error_a, dm_range_error_b;

// Feeder program memory and controller outputs
wire [12:0]  feeder_pm_address;
wire         feeder_pm_wren;
wire [127:0] feeder_instruction;
wire         feeder_controller_start;

// Controller outputs — Feeder handshake
wire         ctrl_controller_idle;
wire         ctrl_clear;

// Controller outputs — Vector_Processor_A control
wire         ctrl_vector_start_a;
wire [2:0]   ctrl_vect_source_a;
wire [2:0]   ctrl_vect_dest_a;
wire [23:0]  ctrl_mem_source_address_a;
wire [23:0]  ctrl_mem_dest_address_a;
wire [15:0]  ctrl_length_a;
wire [3:0]   ctrl_dim0_a;
wire [3:0]   ctrl_dim1_a;

// Controller outputs — VP_A per-layer requantization parameters (MUL writeback)
wire [7:0]   ctrl_scale_a;
wire [7:0]   ctrl_shift_a;

// Controller outputs — Vector_Processor_B control
wire         ctrl_vector_start_b;
wire [2:0]   ctrl_vect_source_b;
wire [2:0]   ctrl_vect_dest_b;
wire [23:0]  ctrl_mem_source_address_b;
wire [15:0]  ctrl_length_b;
wire [3:0]   ctrl_dim0_b;
wire [3:0]   ctrl_dim1_b;

// Controller outputs — Activator and ALU function selects
wire [2:0]   ctrl_activator_funct;
wire [2:0]   ctrl_alu_funct;

// Controller output — Systolic_Array start pulse
wire         ctrl_systolic_array_start;

// Controller output — one-cycle accumulator clear (after a mult writeback),
// independent of the shared inter-operation clear (ctrl_clear).
wire         ctrl_accumulator_clear;

// Controller outputs — active leading dimension (physical row stride) per VP.
wire [23:0]  ctrl_leading_dimension_a;
wire [23:0]  ctrl_leading_dimension_b;

// Vector buffer sclr: asserted when trst is LOW or controller clear is HIGH
wire vb_sclr;
assign vb_sclr = ctrl_clear | ~trst;

// Vector_Processor_A outputs — Data_Memory port a
wire [23:0]  vpa_dm_address;
wire         vpa_dm_wren;
wire [7:0]   vpa_dm_data;
wire [9:0]   vpa_dm_offset;

// Vector_Processor_B outputs — Data_Memory port b
wire [23:0]  vpb_dm_address;
wire         vpb_dm_wren;
wire [7:0]   vpb_dm_data;
wire [9:0]   vpb_dm_offset;

// VP handshake signals
wire         vpa_vector_idle;
wire         vpa_element_valid;
wire         vpb_vector_idle;
wire         vpb_element_valid;

// VP <-> Systolic_Array interface
wire [7:0]   vpa_sa_top_wrreq;
wire [7:0]   vpa_sa_top_data;
wire [7:0]   vpb_sa_left_wrreq;
wire [7:0]   vpb_sa_left_data;
wire [3:0]   vpa_sa_row;
wire [3:0]   vpa_sa_col;

// Systolic_Array outputs
wire         sa_systolic_array_idle;
wire [31:0]  sa_c;

// ALU element_valid enable: both VP_A and VP_B must be valid simultaneously
wire alu_enable;
assign alu_enable = vpa_element_valid & vpb_element_valid;

// Activator element_valid enable
wire act_enable;
assign act_enable = vpa_element_valid;

// Activator and ALU write signals
wire act_write;
wire alu_write;

// Activator and ALU data output signals
wire [7:0] act_data;
wire [7:0] alu_data;

// Vector buffer data/control wires
wire         vb_wrreq;
wire [7:0]   vb_data_in;
assign vb_wrreq   = act_write | alu_write;
assign vb_data_in = alu_write ? alu_data : act_data;

// VP_A vector buffer and downstream data wires
wire         vpa_vb_rdreq;
wire [7:0]   vb_q;
wire [7:0]   vpa_data;    // to Activator and ALU input A
wire [7:0]   vpb_data;    // to ALU input B

assign pm_data    = (program == 1'b0) ? prog_pm_data    : 128'd0;
assign pm_address = (program == 1'b0) ? prog_pm_address : feeder_pm_address;
assign pm_wren    = (program == 1'b0) ? prog_pm_wren    : 1'b0;

// When program=HIGH, VP_A drives the Data_Memory port a
assign dm_data_a    = (program == 1'b0) ? prog_dm_data_a    : vpa_dm_data;
assign dm_address_a = (program == 1'b0) ? prog_dm_address_a : vpa_dm_address;
assign dm_wren_a    = (program == 1'b0) ? prog_dm_wren_a    : vpa_dm_wren;
assign dm_offset_a  = (program == 1'b0) ? prog_dm_offset_a  : vpa_dm_offset;

Programmer prog (
    .i_clk              (i_clk),
    .i_rst              (i_rst),
    .i_mode_select      (i_mode_select),
    .i_input_data_valid (i_input_data_valid),
    .i_device_ready     (i_device_ready),
    .i_input_data       (i_input_data),
    .o_output_data      (o_output_data),
    .o_output_data_valid(o_output_data_valid),
    .i_end_reached      (feeder_end_reached),
    .o_pm_data          (prog_pm_data),
    .o_pm_address       (prog_pm_address),
    .o_pm_wren          (prog_pm_wren),
    .o_dm_data_a        (prog_dm_data_a),
    .o_dm_address_a     (prog_dm_address_a),
    .o_dm_wren_a        (prog_dm_wren_a),
    .o_dm_offset_a      (prog_dm_offset_a),
    .i_dm_q_a           (dm_q_a),
    .o_program          (program),
    .o_tpu_rst          (prog_tpu_rst),
    .i_SPI_Clk          (i_SPI_Clk),
    .o_SPI_MISO         (o_SPI_MISO),
    .i_SPI_MOSI         (i_SPI_MOSI),
    .i_SPI_SS           (i_SPI_SS),
	 
	 .o_error_code(o_error_code)
);

Program_Memory pm (
    .clock  (i_clk),
    .data   (pm_data),
    .address(pm_address),
    .wren   (pm_wren),
    .q      (pm_q)
);

Data_Memory dm (
    .i_clk           (i_clk),
    .i_data_a        (dm_data_a),
    .i_address_a     (dm_address_a),
    .i_wren_a        (dm_wren_a),
    .i_offset_a      (dm_offset_a),
    .o_q_a           (dm_q_a),
    .i_data_b        (vpb_dm_data),
    .i_address_b     (vpb_dm_address),
    .i_wren_b        (vpb_dm_wren),
    .i_offset_b      (vpb_dm_offset),
    .o_q_b           (dm_q_b),
    .o_write_error_a (dm_write_error_a),
    .o_write_error_b (dm_write_error_b),
    .o_range_error_a (dm_range_error_a),
    .o_range_error_b (dm_range_error_b)
);

Feeder feeder (
    .i_clk              (i_clk),
    .i_trst             (trst),
    .i_pm_q             (pm_q),
    .o_pm_address       (feeder_pm_address),
    .o_pm_wren          (feeder_pm_wren),
    .i_controller_idle  (ctrl_controller_idle),
    .o_instruction      (feeder_instruction),
    .o_controller_start (feeder_controller_start),
    .o_end_reached      (feeder_end_reached)
);

Controller ctrl (
    .i_clk                  (i_clk),
    .i_trst                 (trst),
    .i_controller_start     (feeder_controller_start),
    .i_instruction          (feeder_instruction),
    .i_vector_idle_a        (vpa_vector_idle),
    .i_vector_idle_b        (vpb_vector_idle),
    .i_systolic_array_idle  (sa_systolic_array_idle),
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
    .o_scale_a              (ctrl_scale_a),
    .o_shift_a              (ctrl_shift_a),
    .o_vector_start_b       (ctrl_vector_start_b),
    .o_vect_source_b        (ctrl_vect_source_b),
    .o_vect_dest_b          (ctrl_vect_dest_b),
    .o_mem_source_address_b (ctrl_mem_source_address_b),
    .o_length_b             (ctrl_length_b),
    .o_dim0_b               (ctrl_dim0_b),
    .o_dim1_b               (ctrl_dim1_b),
    .o_leading_dimension_a  (ctrl_leading_dimension_a),
    .o_leading_dimension_b  (ctrl_leading_dimension_b),
    .o_activator_funct      (ctrl_activator_funct),
    .o_alu_funct            (ctrl_alu_funct),
    .o_systolic_array_start (ctrl_systolic_array_start),
    .o_accumulator_clear    (ctrl_accumulator_clear)
);

Vector_Processor vp_a (
    .i_clk                 (i_clk),
    .i_trst                (trst),
    .i_vector_start        (ctrl_vector_start_a),
    .i_vect_source         (ctrl_vect_source_a),
    .i_vect_dest           (ctrl_vect_dest_a),
    .i_mem_source_address  (ctrl_mem_source_address_a),
    .i_mem_dest_address    (ctrl_mem_dest_address_a),
    .i_length              (ctrl_length_a),
    .i_dim0                (ctrl_dim0_a),
    .i_dim1                (ctrl_dim1_a),
    .i_leading_dimension   (ctrl_leading_dimension_a),
    .i_scale               (ctrl_scale_a),
    .i_shift               (ctrl_shift_a),
    .o_vector_idle         (vpa_vector_idle),
    .o_element_valid       (vpa_element_valid),
    .o_dm_address          (vpa_dm_address),
    .o_dm_wren             (vpa_dm_wren),
    .o_dm_data             (vpa_dm_data),
    .o_dm_offset           (vpa_dm_offset),
    .i_dm_q                (dm_q_a),
    .o_vb_rdreq            (vpa_vb_rdreq),
    .i_vb_q                (vb_q),
    .o_data                (vpa_data),
    .o_sa_top_data         (vpa_sa_top_data),
    .o_sa_top_wrreq        (vpa_sa_top_wrreq),
    .o_sa_left_data        (),          // VP_A does not write left SA buffers
    .o_sa_left_wrreq       (),
    .o_sa_row              (vpa_sa_row),
    .o_sa_col              (vpa_sa_col),
    .i_sa_c                (sa_c)
);

Vector_Processor vp_b (
    .i_clk                 (i_clk),
    .i_trst                (trst),
    .i_vector_start        (ctrl_vector_start_b),
    .i_vect_source         (ctrl_vect_source_b),
    .i_vect_dest           (ctrl_vect_dest_b),
    .i_mem_source_address  (ctrl_mem_source_address_b),
    .i_mem_dest_address    (24'd0),     // VP_B never writes to memory in current ISA
    .i_length              (ctrl_length_b),
    .i_dim0                (ctrl_dim0_b),
    .i_dim1                (ctrl_dim1_b),
    .i_leading_dimension   (ctrl_leading_dimension_b),
    .i_scale               (8'd0),      // VP_B never requantizes SA outputs
    .i_shift               (8'd0),
    .o_vector_idle         (vpb_vector_idle),
    .o_element_valid       (vpb_element_valid),
    .o_dm_address          (vpb_dm_address),
    .o_dm_wren             (vpb_dm_wren),
    .o_dm_data             (vpb_dm_data),
    .o_dm_offset           (vpb_dm_offset),
    .i_dm_q                (dm_q_b),
    .o_vb_rdreq            (),          // VP_B never reads vector buffer
    .i_vb_q                (8'd0),
    .o_data                (vpb_data),
    .o_sa_top_data         (),          // VP_B does not write top SA buffers
    .o_sa_top_wrreq        (),
    .o_sa_left_data        (vpb_sa_left_data),
    .o_sa_left_wrreq       (vpb_sa_left_wrreq),
    .o_sa_row              (),          // VP_B does not read SA outputs
    .o_sa_col              (),
    .i_sa_c                (32'd0)
);

Vector_Buffer vb (
    .clock (i_clk),
    .data  (vb_data_in),
    .rdreq (vpa_vb_rdreq),
    .sclr  (vb_sclr),
    .wrreq (vb_wrreq),
    .empty (),
    .full  (),
    .q     (vb_q)
);

Activator act (
	.i_clk(i_clk),
	.i_trst(trst),
	.i_enable(act_enable),
	.i_clear(ctrl_clear),
	.o_write(act_write),
	.i_activator_op(ctrl_activator_funct),
	.i_data(vpa_data),
	.o_data(act_data)
);

ALU alu (
	.i_clk(i_clk), //intentionally not connected
	.i_trst(trst),
	.i_enable(alu_enable),
	.i_clear(ctrl_clear),
	.o_write(alu_write),
	.i_alu_op(ctrl_alu_funct),
	.i_data_a(vpa_data),
	.i_data_b(vpb_data),
	.o_data(alu_data)
);

Systolic_Array #(.N(8)) sa (
	.i_clk                  (i_clk),
	.i_trst                 (trst),
	.i_clear                (ctrl_clear),
	.i_accumulator_clear    (ctrl_accumulator_clear),
	.i_systolic_array_start (ctrl_systolic_array_start),
	.o_systolic_array_idle  (sa_systolic_array_idle),
	.i_top_data             (vpa_sa_top_data),
	.i_top_wrreq            (vpa_sa_top_wrreq),
	.i_left_data            (vpb_sa_left_data),
	.i_left_wrreq           (vpb_sa_left_wrreq),
	.i_row                  (vpa_sa_row),
	.i_col                  (vpa_sa_col),
	.o_c                    (sa_c)
);

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
