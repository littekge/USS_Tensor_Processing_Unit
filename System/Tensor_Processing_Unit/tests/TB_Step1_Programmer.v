/*
 * File: TB_Step1_Programmer.v
 * Date: 2026-06-30
 *
 * Testbench for v0.2 Build Step 1: Programmer module (meta-state rewrite).
 * Exercises the new Functional TPU Message Protocol v0.2 (FLASH/INPUT headers,
 * MEM/PROGRAM commands, STOP trailer), the separate program / tpu_rst outputs,
 * and the new device-level IO paths (device input, device output, external
 * output with the device_ready handshake).
 *
 * A behavioral 0x1-buffer model (2-cycle registered read latency, matching the
 * Quartus RAM IP) responds to the Programmer's read/write port so the output
 * meta-states can be verified end to end.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation):
 *
 *   1. cd <path>/Tensor_Processing_Unit
 *   2. Compile:
 *        vlib work
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v
 *        vlog -work work TPU/PROGRAMMER/Programmer.v
 *        vlog -work work tests/TB_Step1_Programmer.v
 *   3. Simulate:
 *        vsim -L altera_mf_ver -voptargs=+acc work.TB_Step1_Programmer -do "add wave -r /*; run -all; quit"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL". A final summary prints pass/fail counts.
 *   All 7 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: Reset defaults — program HIGH, tpu_rst HIGH, output_data_valid LOW.
 *   Test 2: FLASH + MEM — program & tpu_rst go LOW during the session and HIGH
 *           after STOP; MEM bytes land in weight memory (unified address map).
 *   Test 3: PROGRAM — a 128-bit instruction is assembled LSB-first and written
 *           to program memory; a second FLASH overwrites from address 0.
 *   Test 4: DEVICE_INPUT — i_input_data_valid writes i_input_data to 0x1-buffer
 *           address 0 and tpu_rst pulses LOW (program re-run trigger).
 *   Test 5: DEVICE_OUTPUT — i_end_reached emits 0x1-buffer[0] on o_output_data
 *           with a one-cycle o_output_data_valid pulse.
 *   Test 6: EXTERNAL_OUTPUT — external mode emits 10 values from 0x1-buffer[0..9]
 *           paced by the i_device_ready handshake.
 *   Test 7: INPUT header — external mode writes a MEM block to memory; device
 *           mode discards the INPUT transmission (no writes).
 *   Test 8: COM_ERROR — an invalid command code after a FLASH header halts the
 *           FSM in COM_ERROR with no memory writes.
 *   Test 9: WRITE_ERROR — a MEM command targeting ISA address 0x0000 halts the
 *           FSM in WRITE_ERROR with no memory writes.
 *   Test 10: EXTERNAL_INPUT unified routing — an INPUT/MEM block to ISA 0x0005
 *           writes to WEIGHT memory (not just the 0x1 buffer).
 *   Test 11: PROGRAM command inside an INPUT transmission is invalid -> COM_ERROR
 *           (no program-memory writes).
 *   Test 12: IDLE discards stray non-header bytes; a following FLASH still works.
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step1_Programmer;

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    reg          clk;
    reg          rst;

    reg          mode_select;
    reg          input_data_valid;
    reg          device_ready;
    reg  [7:0]   input_data;
    wire [7:0]   output_data;
    wire         output_data_valid;

    reg          end_reached;

    wire [127:0] pm_data;
    wire [9:0]   pm_address;
    wire         pm_wren;
    wire [7:0]   wm_data_a;
    wire [15:0]  wm_address_a;
    wire         wm_wren_a;
    wire [7:0]   buf_data_a;
    wire [15:0]  buf_address_a;
    wire         buf_wren_a;
    wire [7:0]   buf_q_a;
    wire         program;
    wire         tpu_rst;

    // SPI
    reg  spi_clk;
    reg  spi_mosi;
    reg  spi_ss;
    wire spi_miso;

    // -----------------------------------------------------------------------
    // Timing
    // -----------------------------------------------------------------------
    localparam CLK_HALF = 10;   // 50 MHz
    localparam SPI_HALF = 100;  // 5 MHz SPI

    // Protocol codes
    localparam [7:0] FLASH_CODE = 8'h55;
    localparam [7:0] INPUT_CODE = 8'h49;
    localparam [7:0] MEM_CODE   = 8'h4D;
    localparam [7:0] PROG_CODE  = 8'h50;
    localparam [7:0] STOP_CODE  = 8'h53;

    // Programmer error-state encodings (must match Programmer.v)
    localparam [5:0] S_COM_ERROR   = 6'd53;
    localparam [5:0] S_WRITE_ERROR = 6'd54;

    // -----------------------------------------------------------------------
    // DUT
    // -----------------------------------------------------------------------
    Programmer dut (
        .i_clk              (clk),
        .i_rst              (rst),
        .i_mode_select      (mode_select),
        .i_input_data_valid (input_data_valid),
        .i_device_ready     (device_ready),
        .i_input_data       (input_data),
        .o_output_data      (output_data),
        .o_output_data_valid(output_data_valid),
        .i_end_reached      (end_reached),
        .o_pm_data          (pm_data),
        .o_pm_address       (pm_address),
        .o_pm_wren          (pm_wren),
        .o_wm_data_a        (wm_data_a),
        .o_wm_address_a     (wm_address_a),
        .o_wm_wren_a        (wm_wren_a),
        .o_buf_data_a       (buf_data_a),
        .o_buf_address_a    (buf_address_a),
        .o_buf_wren_a       (buf_wren_a),
        .i_buf_q_a          (buf_q_a),
        .o_program          (program),
        .o_tpu_rst          (tpu_rst),
        .i_SPI_Clk          (spi_clk),
        .o_SPI_MISO         (spi_miso),
        .i_SPI_MOSI         (spi_mosi),
        .i_SPI_SS           (spi_ss)
    );

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #CLK_HALF clk = ~clk;

    // -----------------------------------------------------------------------
    // Behavioral 0x1-buffer model (2-cycle registered read latency).
    // Honors the Programmer's writes so DEVICE_INPUT can be read back, and the
    // preloaded contents drive the output meta-states.
    // -----------------------------------------------------------------------
    reg [7:0] buf_mem [0:255];
    reg [7:0] buf_q_stage1, buf_q_reg;
    assign buf_q_a = buf_q_reg;

    always @(posedge clk) begin
        if (buf_wren_a)
            buf_mem[buf_address_a[7:0]] <= buf_data_a;
        buf_q_stage1 <= buf_mem[buf_address_a[7:0]];
        buf_q_reg    <= buf_q_stage1;
    end

    // -----------------------------------------------------------------------
    // Write-event capture
    // -----------------------------------------------------------------------
    integer wm_write_count;
    reg [15:0] wm_addr_log [0:7];
    reg [7:0]  wm_data_log [0:7];

    integer buf_write_count;
    reg [15:0] buf_addr_log [0:7];
    reg [7:0]  buf_data_log [0:7];

    integer pm_write_count;
    reg [9:0]   pm_addr_log [0:3];
    reg [127:0] pm_data_log [0:3];

    always @(posedge clk) begin
        if (wm_wren_a && wm_write_count < 8) begin
            wm_addr_log[wm_write_count] = wm_address_a;
            wm_data_log[wm_write_count] = wm_data_a;
            wm_write_count = wm_write_count + 1;
        end
        if (buf_wren_a && buf_write_count < 8) begin
            buf_addr_log[buf_write_count] = buf_address_a;
            buf_data_log[buf_write_count] = buf_data_a;
            buf_write_count = buf_write_count + 1;
        end
        if (pm_wren && pm_write_count < 4) begin
            pm_addr_log[pm_write_count] = pm_address;
            pm_data_log[pm_write_count] = pm_data;
            pm_write_count = pm_write_count + 1;
        end
    end

    // -----------------------------------------------------------------------
    // Output-event capture (sampled when output_data_valid is HIGH)
    // -----------------------------------------------------------------------
    integer out_count;
    reg [7:0] out_log [0:15];

    always @(posedge clk) begin
        if (output_data_valid && out_count < 16) begin
            out_log[out_count] = output_data;
            out_count = out_count + 1;
        end
    end

    // -----------------------------------------------------------------------
    // Bookkeeping
    // -----------------------------------------------------------------------
    integer pass_count, fail_count;

    task report;
        input pass;
        input [255:0] description;
        begin
            if (pass) begin
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

    // One-cycle pulse on end_reached
    task pulse_end_reached;
    begin
        @(posedge clk); #1;
        end_reached = 1'b1;
        @(posedge clk); #1;
        end_reached = 1'b0;
    end
    endtask

    // One-cycle pulse on input_data_valid with data
    task pulse_input;
        input [7:0] val;
        begin
            @(posedge clk); #1;
            input_data       = val;
            input_data_valid = 1'b1;
            @(posedge clk); #1;
            input_data_valid = 1'b0;
        end
    endtask

    // -----------------------------------------------------------------------
    // Variables
    // -----------------------------------------------------------------------
    integer timeout;
    integer k;
    reg [127:0] expected_instr;

    initial begin
        // Init
        rst              = 0;
        mode_select      = 0;     // device mode
        input_data_valid = 0;
        device_ready     = 0;
        input_data       = 0;
        end_reached      = 0;
        spi_clk          = 0;
        spi_mosi         = 0;
        spi_ss           = 1;
        pass_count       = 0;
        fail_count       = 0;
        wm_write_count   = 0;
        buf_write_count  = 0;
        pm_write_count   = 0;
        out_count        = 0;
        for (k = 0; k < 256; k = k + 1) buf_mem[k] = 8'd0;

        repeat (4) @(posedge clk);
        rst = 1;
        repeat (4) @(posedge clk);

        // ===================================================================
        // Test 1: reset defaults
        // ===================================================================
        $display("Test 1: reset defaults");
        report((program === 1'b1) && (tpu_rst === 1'b1) &&
               (output_data_valid === 1'b0),
               "program HIGH, tpu_rst HIGH, output_data_valid LOW after reset");

        // ===================================================================
        // Test 2: FLASH + MEM -> Weight_Memory; program/tpu_rst toggling
        //   FLASH, MEM @ISA 0x0005 (3 bytes AB CD EF), STOP
        //   ISA addr A -> WM physical A-2, so phys 3,4,5
        // ===================================================================
        $display("Test 2: FLASH + MEM write to Weight_Memory");
        wm_write_count = 0;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h05); // LADD
        send_spi_byte(8'h00); // UADD
        send_spi_byte(8'h03); // LLEN
        send_spi_byte(8'h00); // ULEN
        send_spi_byte(8'hAB);
        send_spi_byte(8'hCD);
        send_spi_byte(8'hEF);
        // Sample program/tpu_rst mid-session (should both be LOW)
        report((program === 1'b0) && (tpu_rst === 1'b0),
               "program LOW and tpu_rst LOW during FLASH session");
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        timeout = 0;
        while (wm_write_count < 3 && timeout < 6000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        report(wm_write_count === 3, "3 writes to WM");
        report(wm_addr_log[0] === 16'd3 && wm_data_log[0] === 8'hAB, "WM phys 3 = 0xAB");
        report(wm_addr_log[2] === 16'd5 && wm_data_log[2] === 8'hEF, "WM phys 5 = 0xEF");
        repeat (10) @(posedge clk);
        report((program === 1'b1) && (tpu_rst === 1'b1),
               "program HIGH and tpu_rst HIGH after STOP");

        // ===================================================================
        // Test 3: PROGRAM instruction write + overwrite on next FLASH
        // ===================================================================
        $display("Test 3: PROGRAM instruction write and overwrite");
        pm_write_count = 0;
        expected_instr = 128'h0F0E0D0C0B0A09080706050403020100;
        // First program: write one instruction (ends at pc 0)
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_spi_byte(PROG_CODE);
        for (k = 0; k < 16; k = k + 1) send_spi_byte(k[7:0]);
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        timeout = 0;
        while (pm_write_count < 1 && timeout < 6000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        report(pm_write_count === 1 && pm_addr_log[0] === 10'd0 &&
               pm_data_log[0] === expected_instr,
               "instruction assembled LSB-first at pm address 0");
        // Second program: a new FLASH session must overwrite from address 0
        pm_write_count = 0;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_spi_byte(PROG_CODE);
        for (k = 0; k < 16; k = k + 1) send_spi_byte(8'hA0 + k[7:0]);
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        timeout = 0;
        while (pm_write_count < 1 && timeout < 6000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        report(pm_write_count === 1 && pm_addr_log[0] === 10'd0,
               "second FLASH overwrites program memory from address 0");

        // ===================================================================
        // Test 4: DEVICE_INPUT
        // ===================================================================
        $display("Test 4: DEVICE_INPUT writes input to 0x1-buffer addr 0");
        mode_select     = 1'b0; // device mode
        buf_write_count = 0;
        pulse_input(8'h5A);
        timeout = 0;
        while (buf_write_count < 1 && timeout < 2000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        report(buf_write_count === 1 && buf_addr_log[0] === 16'd0 &&
               buf_data_log[0] === 8'h5A,
               "input 0x5A written to 0x1-buffer address 0");

        // ===================================================================
        // Test 5: DEVICE_OUTPUT
        //   Preload buffer[0]; pulse end_reached; expect one output of buf[0].
        // ===================================================================
        $display("Test 5: DEVICE_OUTPUT emits 0x1-buffer[0]");
        mode_select = 1'b0;
        // Settle past any pending buffer write from Test 4 before preloading.
        repeat (3) @(posedge clk); #1;
        buf_mem[0]  = 8'h77;
        out_count   = 0;
        pulse_end_reached;
        timeout = 0;
        while (out_count < 1 && timeout < 2000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        repeat (5) @(posedge clk);
        report(out_count === 1 && out_log[0] === 8'h77,
               "single output 0x77 with output_data_valid pulse");

        // ===================================================================
        // Test 6: EXTERNAL_OUTPUT (10 values with device_ready handshake)
        // ===================================================================
        $display("Test 6: EXTERNAL_OUTPUT emits 10 values");
        mode_select = 1'b1; // external mode
        for (k = 0; k < 10; k = k + 1) buf_mem[k] = 8'hC0 + k[7:0];
        out_count = 0;
        pulse_end_reached;
        // Service the handshake: pulse device_ready whenever the DUT is waiting.
        timeout = 0;
        while (out_count < 10 && timeout < 8000) begin
            @(posedge clk); #1;
            if (dut.S == 6'd49) begin // X_OUT_WAIT_RDY
                device_ready = 1'b1;
                @(posedge clk); #1;
                device_ready = 1'b0;
            end
            timeout = timeout + 1;
        end
        report(out_count === 10, "exactly 10 values emitted");
        report(out_log[0] === 8'hC0 && out_log[9] === 8'hC9,
               "first value 0xC0, tenth value 0xC9");

        // ===================================================================
        // Test 7: INPUT header — external writes, device-mode discards
        // ===================================================================
        $display("Test 7: INPUT header routing by mode");
        // External mode: INPUT + MEM should write to the 0x1 buffer (ISA 0x1).
        mode_select     = 1'b1;
        buf_write_count = 0;
        spi_ss = 0;
        send_spi_byte(INPUT_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h01); // LADD = ISA 0x0001 (0x1 buffer)
        send_spi_byte(8'h00); // UADD
        send_spi_byte(8'h01); // LLEN = 1
        send_spi_byte(8'h00); // ULEN
        send_spi_byte(8'h3C); // data
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        timeout = 0;
        while (buf_write_count < 1 && timeout < 6000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        report(buf_write_count === 1 && buf_addr_log[0] === 16'd0 &&
               buf_data_log[0] === 8'h3C,
               "external INPUT writes 0x3C to 0x1-buffer address 0");
        // Device mode: the same INPUT transmission must be discarded (no writes).
        mode_select     = 1'b0;
        buf_write_count = 0;
        wm_write_count  = 0;
        spi_ss = 0;
        send_spi_byte(INPUT_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h01);
        send_spi_byte(8'h00);
        send_spi_byte(8'h01);
        send_spi_byte(8'h00);
        send_spi_byte(8'h99);
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        repeat (400) @(posedge clk);
        report((buf_write_count === 0) && (wm_write_count === 0),
               "device-mode INPUT transmission discarded (no writes)");

        // ===================================================================
        // Test 8: COM_ERROR on an invalid command code after a FLASH header
        // ===================================================================
        $display("Test 8: COM_ERROR on invalid command code");
        rst = 0; repeat (3) @(posedge clk); rst = 1; repeat (3) @(posedge clk);
        mode_select     = 1'b0;
        wm_write_count  = 0;
        buf_write_count = 0;
        pm_write_count  = 0;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_spi_byte(8'hFF);   // invalid command code
        spi_ss = 1;
        repeat (300) @(posedge clk);
        report((dut.S === S_COM_ERROR) && (wm_write_count === 0) &&
               (buf_write_count === 0) && (pm_write_count === 0),
               "COM_ERROR, no writes on invalid command code");

        // ===================================================================
        // Test 9: WRITE_ERROR on a MEM command targeting ISA address 0x0000
        // ===================================================================
        $display("Test 9: WRITE_ERROR on MEM to address 0x0000");
        rst = 0; repeat (3) @(posedge clk); rst = 1; repeat (3) @(posedge clk);
        mode_select     = 1'b0;
        wm_write_count  = 0;
        buf_write_count = 0;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h00); // LADD = 0
        send_spi_byte(8'h00); // UADD = 0  -> ISA address 0x0000 (illegal)
        send_spi_byte(8'h01); // LLEN = 1
        send_spi_byte(8'h00); // ULEN
        send_spi_byte(8'hAA); // data (must NOT be written)
        spi_ss = 1;
        repeat (300) @(posedge clk);
        report((dut.S === S_WRITE_ERROR) && (wm_write_count === 0) &&
               (buf_write_count === 0),
               "WRITE_ERROR, no writes on MEM to address 0");

        // ===================================================================
        // Test 10: EXTERNAL_INPUT unified routing -> weight memory
        //   external mode, INPUT + MEM @ISA 0x0005 (WM phys 3), 2 bytes.
        // ===================================================================
        $display("Test 10: EXTERNAL_INPUT MEM routes to weight memory");
        rst = 0; repeat (3) @(posedge clk); rst = 1; repeat (3) @(posedge clk);
        mode_select    = 1'b1; // external mode
        wm_write_count = 0;
        spi_ss = 0;
        send_spi_byte(INPUT_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h05); // LADD = ISA 0x0005
        send_spi_byte(8'h00); // UADD
        send_spi_byte(8'h02); // LLEN = 2
        send_spi_byte(8'h00); // ULEN
        send_spi_byte(8'h77);
        send_spi_byte(8'h88);
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        timeout = 0;
        while (wm_write_count < 2 && timeout < 6000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        report((wm_write_count === 2) &&
               (wm_addr_log[0] === 16'd3) && (wm_data_log[0] === 8'h77) &&
               (wm_addr_log[1] === 16'd4) && (wm_data_log[1] === 8'h88),
               "external INPUT writes 0x77/0x88 to WM phys 3/4");

        // ===================================================================
        // Test 11: PROGRAM command inside an INPUT transmission -> COM_ERROR
        // ===================================================================
        $display("Test 11: PROGRAM command in INPUT transmission -> COM_ERROR");
        rst = 0; repeat (3) @(posedge clk); rst = 1; repeat (3) @(posedge clk);
        mode_select    = 1'b1; // external mode
        pm_write_count = 0;
        spi_ss = 0;
        send_spi_byte(INPUT_CODE);
        send_spi_byte(PROG_CODE);  // invalid command for an INPUT session
        spi_ss = 1;
        repeat (300) @(posedge clk);
        report((dut.S === S_COM_ERROR) && (pm_write_count === 0),
               "COM_ERROR, no program-memory writes");

        // ===================================================================
        // Test 12: IDLE discards stray non-header bytes before a valid FLASH
        // ===================================================================
        $display("Test 12: stray non-header bytes discarded; following FLASH works");
        rst = 0; repeat (3) @(posedge clk); rst = 1; repeat (3) @(posedge clk);
        mode_select    = 1'b0;
        wm_write_count = 0;
        spi_ss = 0;
        send_spi_byte(8'h00);   // junk (not a header)
        send_spi_byte(8'hFF);   // junk
        send_spi_byte(8'h4D);   // junk (MEM code outside a session -> discarded)
        send_spi_byte(FLASH_CODE);
        send_spi_byte(MEM_CODE);
        send_spi_byte(8'h05);   // ISA 0x0005 -> WM phys 3
        send_spi_byte(8'h00);
        send_spi_byte(8'h01);   // LLEN = 1
        send_spi_byte(8'h00);
        send_spi_byte(8'h5C);
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        timeout = 0;
        while (wm_write_count < 1 && timeout < 6000) begin
            @(posedge clk); timeout = timeout + 1;
        end
        report((wm_write_count === 1) && (wm_addr_log[0] === 16'd3) &&
               (wm_data_log[0] === 8'h5C),
               "FLASH after junk writes 0x5C to WM phys 3");

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
