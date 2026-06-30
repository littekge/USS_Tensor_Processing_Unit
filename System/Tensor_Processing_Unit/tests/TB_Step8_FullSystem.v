/*
 * File: TB_Step8_FullSystem.v
 * Date: 2026-06-30
 *
 * Full-system integration test for the v0.2 TPU. Drives the top-level TPU
 * module exactly as real hardware would, exercising BOTH improved-IO modes of
 * the v0.2 Programmer end to end:
 *
 *   DEVICE MODE (mode_select = 0):
 *     A FLASH session programs a self-contained ReLU program
 *     (relu 0x0001 -> 0x0001, len 1) that activates the single 0x1-buffer value.
 *     A device input is supplied via i_input_data / i_input_data_valid; the
 *     Programmer writes it to the 0x1 buffer, pulses tpu_rst so the program
 *     re-runs, and on end emits the activated value on o_output_data with a
 *     one-cycle o_output_data_valid pulse.
 *
 *   EXTERNAL MODE (mode_select = 1):
 *     A FLASH session programs relu 0x0001 -> 0x0001 over len 10. An INPUT
 *     transmission (Message Protocol v0.2) writes a 10-element vector to the
 *     0x1 buffer; the program re-runs and, on end, the Programmer emits all 10
 *     activated values, pacing each with the i_device_ready handshake.
 *
 * The program reads from and writes to the 0x1 buffer, so the Programmer's
 * output meta-states read back exactly what the program computed via the real
 * RAM IP (true end-to-end; no internal RAM array referenced).
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation):
 *
 *   1. cd <path>/Tensor_Processing_Unit
 *   2. Compile the full hierarchy (altera_mf required for the RAM/FIFO IP):
 *        vlib work
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v
 *        vlog -work work TPU/PROGRAMMER/PROGRAM_MEMORY/Program_Memory.v
 *        vlog -work work TPU/PROGRAMMER/Programmer.v
 *        vlog -work work TPU/PROGRAMMER/Feeder.v
 *        vlog -work work TPU/CONTROL/Controller.v
 *        vlog -work work TPU/MEMORY/WEIGHT_MEMORY/Weight_Memory.v
 *        vlog -work work TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v
 *        vlog -work work TPU/MEMORY/VECTOR_BUFFER/Vector_Buffer.v
 *        vlog -work work TPU/PROCESSING/Activator.v
 *        vlog -work work TPU/PROCESSING/ALU.v
 *        vlog -work work TPU/PROCESSING/Vector_Processor.v
 *        vlog -work work TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v
 *        vlog -work work TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v
 *        vlog -work work TPU/SYSTOLIC_ARRAY/Systolic_Array.v
 *        vlog -work work TPU/TPU.v
 *        vlog -work work tests/TB_Step8_FullSystem.v
 *   3. Simulate (do NOT optimize signals away; log all waveforms):
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step8_FullSystem -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL". A final summary prints pass/fail counts.
 *   All 7 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: Device-mode FLASH completes — program/tpu_rst return HIGH after STOP.
 *   Test 2: Device input (positive) — relu pass-through: input 42 -> output 42.
 *   Test 3: Device input (negative) — relu clamps: input -9 -> output 0.
 *   Test 4: External-mode re-FLASH completes (len-10 relu program loaded).
 *   Test 5: External output emits exactly 10 values with the device_ready
 *           handshake.
 *   Test 6: External output values equal relu(inputs) (pass-through + clamp).
 *   Test 7: Clean completion — Controller IDLE, Feeder DONE, no fault states.
 *   Test 8: External re-input without re-FLASH — a second INPUT transmission
 *           re-runs the resident program and emits 10 new relu outputs
 *           (steady-state operating loop).
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step8_FullSystem;

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    reg  clk;
    reg  rst;

    reg  spi_clk;
    reg  spi_mosi;
    reg  spi_ss;
    wire spi_miso;
    wire [31:0] debug_val;

    reg         mode_select;
    reg         input_data_valid;
    reg  [7:0]  input_data;
    wire [7:0]  output_data;
    wire        output_data_valid;

    // -----------------------------------------------------------------------
    // Timing / constants
    // -----------------------------------------------------------------------
    localparam CLK_HALF = 10;   // 50 MHz
    localparam SPI_HALF = 100;  // 5 MHz SPI

    localparam [7:0] FLASH_CODE = 8'h55;
    localparam [7:0] INPUT_CODE = 8'h49;
    localparam [7:0] STOP_CODE  = 8'h53;
    localparam [7:0] MEM_CODE   = 8'h4D;
    localparam [7:0] PROG_CODE  = 8'h50;

    // Observed FSM state encodings (must match the DUT modules)
    localparam [3:0] FEEDER_DONE       = 4'd7;   // Feeder.v DONE
    localparam [3:0] CTRL_IDLE         = 4'd2;   // Controller.v IDLE
    localparam [5:0] PROG_X_OUT_WAITRDY = 6'd49; // Programmer.v X_OUT_WAIT_RDY

    // -----------------------------------------------------------------------
    // device_ready auto-handshake: assert whenever the Programmer is waiting in
    // EXTERNAL_OUTPUT for the consumer to accept the next value.
    // -----------------------------------------------------------------------
    wire device_ready;
    assign device_ready = (dut.prog.S == PROG_X_OUT_WAITRDY);

    // -----------------------------------------------------------------------
    // DUT
    // -----------------------------------------------------------------------
    TPU dut (
        .i_clk               (clk),
        .i_rst               (rst),
        .i_SPI_Clk           (spi_clk),
        .o_SPI_MISO          (spi_miso),
        .i_SPI_MOSI          (spi_mosi),
        .i_SPI_SS            (spi_ss),
        .i_mode_select       (mode_select),
        .i_input_data_valid  (input_data_valid),
        .i_device_ready      (device_ready),
        .i_input_data        (input_data),
        .o_output_data       (output_data),
        .o_output_data_valid (output_data_valid),
        .o_debug_val         (debug_val)
    );

    initial clk = 0;
    always #CLK_HALF clk = ~clk;

    // -----------------------------------------------------------------------
    // Output capture (sampled when output_data_valid is HIGH)
    // -----------------------------------------------------------------------
    integer out_count;
    reg [7:0] out_log [0:31];

    always @(posedge clk) begin
        if (output_data_valid && out_count < 32) begin
            out_log[out_count] = output_data;
            out_count = out_count + 1;
        end
    end

    // -----------------------------------------------------------------------
    // Bookkeeping
    // -----------------------------------------------------------------------
    integer pass_count, fail_count;

    task check;
        input         ok;
        input [255:0] description;
        begin
            if (ok) begin
                $display("  PASS: %0s", description);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: %0s", description);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // SPI byte (mode 0, MSB first)
    task send_spi_byte;
        input [7:0] byte_val;
        integer i;
        begin
            for (i = 7; i >= 0; i = i - 1) begin
                spi_mosi = byte_val[i];
                #SPI_HALF;
                spi_clk = 1;
                #SPI_HALF;
                spi_clk = 0;
            end
        end
    endtask

    // Send a 128-bit instruction as a PROGRAM command (LSB byte first)
    task send_program;
        input [127:0] instr;
        integer i;
        begin
            send_spi_byte(PROG_CODE);
            for (i = 0; i < 16; i = i + 1)
                send_spi_byte(instr[i*8 +: 8]);
        end
    endtask

    // ReLU instruction builder (opcode 1001 | rs1 | rd | len | pad | funct3)
    function [127:0] mk_relu;
        input [15:0] rs1;
        input [15:0] rd;
        input [15:0] len;
        begin
            mk_relu = {4'b1001, rs1, rd, len, 73'd0, 3'd0};
        end
    endfunction

    // -----------------------------------------------------------------------
    // Wait helpers
    // -----------------------------------------------------------------------
    integer wc;
    task wait_feeder_done;
        input integer max_cycles;
        begin
            wc = 0;
            while ((dut.feeder.S !== FEEDER_DONE) && (wc < max_cycles)) begin
                @(posedge clk); wc = wc + 1;
            end
        end
    endtask

    // Wait until out_count reaches a target (or timeout)
    task wait_outputs;
        input integer target;
        input integer max_cycles;
        begin
            wc = 0;
            while ((out_count < target) && (wc < max_cycles)) begin
                @(posedge clk); wc = wc + 1;
            end
        end
    endtask

    // -----------------------------------------------------------------------
    // External-mode input vector and its expected relu output
    // -----------------------------------------------------------------------
    reg [7:0] ext_in  [0:9];
    reg [7:0] ext_exp [0:9];
    integer m;
    integer ext_ok;

    // -----------------------------------------------------------------------
    // Main
    // -----------------------------------------------------------------------
    initial begin
        rst              = 0;
        spi_clk          = 0;
        spi_mosi         = 0;
        spi_ss           = 1;
        mode_select      = 0;     // device mode
        input_data_valid = 0;
        input_data       = 0;
        pass_count       = 0;
        fail_count       = 0;
        out_count        = 0;

        repeat (4) @(posedge clk);
        rst = 1;
        repeat (4) @(posedge clk);

        // ===================================================================
        // DEVICE MODE
        // ===================================================================
        mode_select = 1'b0;

        // FLASH a 1-element relu program: relu 0x0001 -> 0x0001, len 1, then end.
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_program(mk_relu(16'h0001, 16'h0001, 16'd1));
        send_program(128'd0); // end
        send_spi_byte(STOP_CODE);
        spi_ss = 1;

        // The program auto-runs once on the initial 0x1-buffer contents.
        wait_feeder_done(200000);
        repeat (60) @(posedge clk);

        $display("Test 1: device-mode FLASH completes");
        check((dut.program === 1'b1) && (dut.prog.o_tpu_rst === 1'b1) &&
              (dut.feeder.S === FEEDER_DONE),
              "program HIGH, tpu_rst HIGH, Feeder DONE after FLASH");

        // ---- Test 2: positive device input passes through relu -------------
        $display("Test 2: device input 42 -> relu -> output 42");
        out_count = 0;
        @(posedge clk); #1;
        input_data       = 8'd42;
        input_data_valid = 1'b1;
        @(posedge clk); #1;
        input_data_valid = 1'b0;
        wait_outputs(1, 200000);
        repeat (20) @(posedge clk);
        check((out_count >= 1) && (out_log[0] === 8'd42),
              "device output equals 42");

        // ---- Test 3: negative device input clamps to 0 ---------------------
        $display("Test 3: device input -9 -> relu -> output 0");
        out_count = 0;
        @(posedge clk); #1;
        input_data       = 8'hF7;  // -9
        input_data_valid = 1'b1;
        @(posedge clk); #1;
        input_data_valid = 1'b0;
        wait_outputs(1, 200000);
        repeat (20) @(posedge clk);
        check((out_count >= 1) && (out_log[0] === 8'd0),
              "device output equals 0 (relu clamp)");

        // ===================================================================
        // EXTERNAL MODE
        // ===================================================================
        mode_select = 1'b1;

        // Re-FLASH a 10-element relu program: relu 0x0001 -> 0x0001, len 10.
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_program(mk_relu(16'h0001, 16'h0001, 16'd10));
        send_program(128'd0); // end
        send_spi_byte(STOP_CODE);
        spi_ss = 1;

        // Auto-run + the initial EXTERNAL_OUTPUT (10 values) drains here.
        wait_feeder_done(200000);
        repeat (40) @(posedge clk);
        wait_outputs(10, 200000);   // let the initial 10-value output complete
        repeat (40) @(posedge clk);

        $display("Test 4: external-mode re-FLASH completes");
        check((dut.program === 1'b1) && (dut.feeder.S === FEEDER_DONE),
              "program HIGH and Feeder DONE after external FLASH");

        // Define the input vector and its expected relu output.
        ext_in[0]=8'd10;  ext_in[1]=8'hFD; ext_in[2]=8'd0;   ext_in[3]=8'd127;
        ext_in[4]=8'h80;  ext_in[5]=8'd5;  ext_in[6]=8'hCE;  ext_in[7]=8'd100;
        ext_in[8]=8'hFF;  ext_in[9]=8'd64;
        for (m = 0; m < 10; m = m + 1)
            ext_exp[m] = ($signed(ext_in[m]) > 0) ? ext_in[m] : 8'd0;

        // ---- INPUT transmission: write the 10-element vector to 0x1 buffer -
        out_count = 0;
        spi_ss = 0;
        send_spi_byte(INPUT_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h01);   // LADD = ISA 0x0001 (0x1 buffer)
        send_spi_byte(8'h00);   // UADD
        send_spi_byte(8'd10);   // LLEN = 10
        send_spi_byte(8'd0);    // ULEN
        for (m = 0; m < 10; m = m + 1) send_spi_byte(ext_in[m]);
        send_spi_byte(STOP_CODE);
        spi_ss = 1;

        // Program re-runs (relu over 10 elements), then EXTERNAL_OUTPUT emits 10.
        wait_outputs(10, 400000);
        repeat (40) @(posedge clk);

        $display("Test 5: external output emits 10 values");
        check(out_count === 10, "exactly 10 values emitted with device_ready handshake");

        $display("Test 6: external output values equal relu(inputs)");
        ext_ok = 1;
        for (m = 0; m < 10; m = m + 1)
            if (out_log[m] !== ext_exp[m]) ext_ok = 0;
        $display("  out: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 out_log[0], out_log[1], out_log[2], out_log[3], out_log[4],
                 out_log[5], out_log[6], out_log[7], out_log[8], out_log[9]);
        $display("  exp: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 ext_exp[0], ext_exp[1], ext_exp[2], ext_exp[3], ext_exp[4],
                 ext_exp[5], ext_exp[6], ext_exp[7], ext_exp[8], ext_exp[9]);
        check(ext_ok === 1, "all 10 outputs equal relu(inputs)");

        // ---- Test 7: clean completion --------------------------------------
        $display("Test 7: clean completion (no fault states)");
        check((dut.ctrl.S === CTRL_IDLE) && (dut.feeder.S === FEEDER_DONE) &&
              (dut.program === 1'b1),
              "Controller IDLE, Feeder DONE, program HIGH at end");

        // ===================================================================
        // Test 8: external re-input without re-FLASH (steady-state loop).
        //   Send a second INPUT vector (all positive) to the resident len-10
        //   relu program; expect 10 new pass-through outputs [1..10].
        // ===================================================================
        $display("Test 8: external re-input re-runs resident program");
        for (m = 0; m < 10; m = m + 1) ext_in[m] = m + 1; // 1..10 (all positive)
        out_count = 0;
        spi_ss = 0;
        send_spi_byte(INPUT_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h01);   // ISA 0x0001 (0x1 buffer)
        send_spi_byte(8'h00);
        send_spi_byte(8'd10);   // LLEN = 10
        send_spi_byte(8'd0);
        for (m = 0; m < 10; m = m + 1) send_spi_byte(ext_in[m]);
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        wait_outputs(10, 400000);
        repeat (40) @(posedge clk);
        ext_ok = 1;
        if (out_count !== 10) ext_ok = 0;
        for (m = 0; m < 10; m = m + 1)
            if (out_log[m] !== ext_in[m]) ext_ok = 0;
        $display("  out: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 out_log[0], out_log[1], out_log[2], out_log[3], out_log[4],
                 out_log[5], out_log[6], out_log[7], out_log[8], out_log[9]);
        check(ext_ok === 1, "second INPUT re-runs program -> outputs [1..10]");

        // ===================================================================
        // Summary
        // ===================================================================
        $display("");
        $display("================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("================================");
        $finish;
    end

endmodule
