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
 *        vlog -work work TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v
 *        vlog -work work TPU/MEMORY/MEM_UNIT/Mem_Unit.v
 *        vlog -work work TPU/MEMORY/Data_Memory.v
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
 *   All 13 tests should show PASS for a correct implementation.
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
 *   Test 9: FLASH weights via MEM + a single mult against them — verifies the
 *           weight-flash + systolic-array mult path end to end (the path a
 *           neural-network program exercises; previously uncovered in v0.2).
 *   Test 10: Two consecutive mult instructions in one program — the second
 *           mult's result must match the first (identical operands), proving
 *           the MAC accumulators are cleared between operations rather than
 *           accumulating stale products into saturation.
 *   Test 11: Signed mult (v0.2.1) — a mult with NEGATIVE operands streamed
 *           through the systolic array requantizes to negative results
 *           ([[-1,2],[3,-4]]). Fails on the old unsigned MAC multiply, which
 *           reads a negative int8 (e.g. 0xF0 = -16) as +240 and produces a
 *           large positive product instead.
 *   Test 12: Per-layer dyadic requant (v0.3) — a mult carrying M0=3, n=4
 *           requantizes C = [[100,50],[25,10]] to [[19,9],[5,2]] via
 *           clamp((3x + 8) >> 4), exercising the runtime M0/n datapath.
 *   Test 13 (v0.5): Tiled matmul with leading-dimension addressing. C(2x2) =
 *           A(2x16) * B(16x2) with contraction K=16 tiled into two K=8 passes
 *           (a multip accumulating the first tile, terminated by a mult that
 *           adds the second tile, requantizes, and clears). A stride sets
 *           ld1=16 so each 2x8 sub-block of A is read in place (row 1 at
 *           base+16, not base+8); ld2=2 addresses B/dest. A=[row0=1s,row1=2s],
 *           B[k]=[1,2] -> C=[[16,32],[32,64]], requant (M0=1,n=4) = [[1,2],[2,4]].
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
        .o_end_reached       (),                // exposed at TPU boundary; unused here
        .o_debug_val         (debug_val)
    );

    // v0.3: requantization is per-layer and supplied at runtime via each MUL
    // instruction's M0/n fields (no compile-time REQUANT_SHIFT parameter). The
    // mk_mul builder embeds M0=1, n=8 by default so the compute tests (9-11)
    // keep their pre-v0.3 hand-computed values; Test 12 exercises a non-trivial
    // M0/n end to end.

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
    // Data_Memory shadow — mirrors the muxed Data_Memory port-a writes (both the
    // Programmer's flashed data and the Vector_Processor's compute write-backs
    // flow through this port). v0.4 flat unified addressing: a write to flat
    // address 0x1 targets the 0x1 buffer (indexed by the offset port); any other
    // non-zero flat address targets main data memory (indexed by the flat
    // address). Lets the compute tests read back the requantized mult result
    // without referencing any vendor-IP internal array.
    // -----------------------------------------------------------------------
    reg [7:0] shadow_main [0:1023];
    reg [7:0] shadow_buf  [0:1023];
    integer si;

    always @(posedge clk) begin
        if (dut.dm_wren_a) begin
            if (dut.dm_address_a == 24'd1)
                shadow_buf[dut.dm_offset_a] <= dut.dm_data_a;
            else if (dut.dm_address_a != 24'd0)
                shadow_main[dut.dm_address_a[9:0]] <= dut.dm_data_a;
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

    // ReLU instruction builder — ISA v0.4 ACT format:
    //   opcode[127:124] | rs1[123:100] | rd[99:76] | len[75:60] | pad[59:0]
    function [127:0] mk_relu;
        input [23:0] rs1;
        input [23:0] rd;
        input [15:0] len;
        begin
            mk_relu = {4'b1001, rs1, rd, len, 60'd0};
        end
    endfunction

    // mult instruction builder with explicit per-layer requant parameters —
    // ISA v0.4 MUL format:
    //   opcode[127:124] | rs1[123:100] | sz11[99:96] | sz12[95:92] |
    //   rs2[91:68] | sz21[67:64] | sz22[63:60] | rd[59:36] |
    //   M0[35:28] | n[27:20] | pad[19:0]
    function [127:0] mk_mul_rq;
        input [23:0] rs1;
        input [3:0]  sz11, sz12;
        input [23:0] rs2;
        input [3:0]  sz21, sz22;
        input [23:0] rd;
        input [7:0]  scale, shift;
        begin
            mk_mul_rq = {4'b1000, rs1, sz11, sz12, rs2, sz21, sz22, rd,
                         scale, shift, 20'd0};
        end
    endfunction

    // Default mult builder: M0=1, n=8 reproduces the pre-v0.3 fixed >>8
    // requantization so the existing compute tests keep their expected results.
    function [127:0] mk_mul;
        input [23:0] rs1;
        input [3:0]  sz11, sz12;
        input [23:0] rs2;
        input [3:0]  sz21, sz22;
        input [23:0] rd;
        begin
            mk_mul = mk_mul_rq(rs1, sz11, sz12, rs2, sz21, sz22, rd, 8'd1, 8'd8);
        end
    endfunction

    // MUL builder with an explicit funct3 (ISA v0.5): mk_mul_rq embeds funct3=0
    // (mult); mk_multip embeds funct3=1 (multip — does not clear the
    // accumulator after writeback, so a following MUL accumulates into it).
    function [127:0] mk_mul_f3;
        input [23:0] rs1;
        input [3:0]  sz11, sz12;
        input [23:0] rs2;
        input [3:0]  sz21, sz22;
        input [23:0] rd;
        input [7:0]  scale, shift;
        input [2:0]  f3;
        begin
            mk_mul_f3 = {4'b1000, rs1, sz11, sz12, rs2, sz21, sz22, rd,
                         scale, shift, 17'd0, f3};
        end
    endfunction

    function [127:0] mk_multip;
        input [23:0] rs1;
        input [3:0]  sz11, sz12;
        input [23:0] rs2;
        input [3:0]  sz21, sz22;
        input [23:0] rd;
        input [7:0]  scale, shift;
        begin
            mk_multip = mk_mul_f3(rs1, sz11, sz12, rs2, sz21, sz22, rd,
                                  scale, shift, 3'h1);
        end
    endfunction

    // stride builder (ISA v0.5 CONFIG format): opcode 1111 | im1[123:100] |
    // im2[99:76] | reserved[75:3] | funct3[2:0]=0. im1 -> ld1, im2 -> ld2.
    function [127:0] mk_stride;
        input [23:0] im1, im2;
        begin
            mk_stride = {4'b1111, im1, im2, 76'd0};
        end
    endfunction

    // Send a 4-byte MEM command block to a 24-bit flat ISA address (Message
    // Protocol v0.3: 3-byte little-endian address LADD/MADD/UADD, then length).
    task send_mem4;
        input [23:0] addr;
        input [7:0]  b0, b1, b2, b3;
        begin
            send_spi_byte(MEM_CODE);
            send_spi_byte(addr[7:0]);    // LADD
            send_spi_byte(addr[15:8]);   // MADD
            send_spi_byte(addr[23:16]);  // UADD
            send_spi_byte(8'd4);   // LLEN = 4
            send_spi_byte(8'd0);   // ULEN = 0
            send_spi_byte(b0);
            send_spi_byte(b1);
            send_spi_byte(b2);
            send_spi_byte(b3);
        end
    endtask

    // Send a MEM command block of `count` bytes (count <= 255) from send_buf,
    // to a 24-bit flat ISA address (LLEN=count, ULEN=0). Used by Test 13 to
    // flash the parent matrices for the tiled matmul.
    reg [7:0] send_buf [0:255];
    task send_mem_block;
        input [23:0] addr;
        input integer count;
        integer i;
        begin
            send_spi_byte(MEM_CODE);
            send_spi_byte(addr[7:0]);    // LADD
            send_spi_byte(addr[15:8]);   // MADD
            send_spi_byte(addr[23:16]);  // UADD
            send_spi_byte(count[7:0]);   // LLEN
            send_spi_byte(8'd0);         // ULEN
            for (i = 0; i < count; i = i + 1)
                send_spi_byte(send_buf[i]);
        end
    endtask

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
        send_program(mk_relu(24'h000001, 24'h000001, 16'd1));
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
        send_program(mk_relu(24'h000001, 24'h000001, 16'd10));
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
        send_spi_byte(8'h01);   // LADD = ISA 0x000001 (0x1 buffer)
        send_spi_byte(8'h00);   // MADD
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
        send_spi_byte(8'h01);   // LADD = ISA 0x000001 (0x1 buffer)
        send_spi_byte(8'h00);   // MADD
        send_spi_byte(8'h00);   // UADD
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
        // Test 9: FLASH weights via MEM + a single mult against them.
        //   Verifies the weight-flash + systolic-array mult path end to end.
        //   v0.4 flat unified addressing (physical addr == ISA addr):
        //     rs1 @0x000002 = [[16,0],[0,16]]   -> flat 2..5
        //     rs2 @0x000006 = [[16,32],[48,64]] -> flat 6..9
        //     rd  @0x00000A                      -> flat 10..13
        //   Products 256/512/768/1024 requantized (>>8) -> [[1,2],[3,4]].
        // ===================================================================
        $display("Test 9: FLASH weights via MEM + single mult (coverage gap #3)");
        rst = 0; repeat (4) @(posedge clk); rst = 1; repeat (4) @(posedge clk);
        mode_select = 1'b0;
        for (si = 0; si < 1024; si = si + 1) shadow_main[si] = 8'hEE;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_mem4(24'h000002, 8'd16, 8'd0,  8'd0,  8'd16); // rs1
        send_mem4(24'h000006, 8'd16, 8'd32, 8'd48, 8'd64); // rs2
        send_program(mk_mul(24'h000002, 4'd2, 4'd2, 24'h000006, 4'd2, 4'd2, 24'h00000A));
        send_program(128'd0); // end
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        wait_feeder_done(200000);
        repeat (60) @(posedge clk);
        $display("  mult result mem[10..13] = %0d %0d %0d %0d  (expect 1 2 3 4)",
                 $signed(shadow_main[10]), $signed(shadow_main[11]),
                 $signed(shadow_main[12]), $signed(shadow_main[13]));
        check((shadow_main[10] === 8'd1) && (shadow_main[11] === 8'd2) &&
              (shadow_main[12] === 8'd3) && (shadow_main[13] === 8'd4),
              "single mult from flashed weights = [1,2,3,4]");

        // ===================================================================
        // Test 10: two consecutive mult instructions in one program.
        //   Both mults use the SAME operands and so must produce the SAME
        //   result. mult1 -> rdA @0x00000A (flat 10..13); mult2 -> rdB @0x00000E
        //   (flat 14..17). If the MAC accumulators are not cleared between the
        //   two mults, mult2 accumulates on top of mult1 (doubling to
        //   [[2,4],[6,8]] or saturating) and this test fails.
        // ===================================================================
        $display("Test 10: two consecutive mults — MAC clear between ops (possibility #2)");
        rst = 0; repeat (4) @(posedge clk); rst = 1; repeat (4) @(posedge clk);
        mode_select = 1'b0;
        for (si = 0; si < 1024; si = si + 1) shadow_main[si] = 8'hEE;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_mem4(24'h000002, 8'd16, 8'd0,  8'd0,  8'd16); // rs1
        send_mem4(24'h000006, 8'd16, 8'd32, 8'd48, 8'd64); // rs2
        send_program(mk_mul(24'h000002, 4'd2, 4'd2, 24'h000006, 4'd2, 4'd2, 24'h00000A)); // mult1 -> flat 10..13
        send_program(mk_mul(24'h000002, 4'd2, 4'd2, 24'h000006, 4'd2, 4'd2, 24'h00000E)); // mult2 -> flat 14..17
        send_program(128'd0); // end
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        wait_feeder_done(200000);
        repeat (80) @(posedge clk);
        $display("  mult1 mem[10..13] = %0d %0d %0d %0d",
                 $signed(shadow_main[10]), $signed(shadow_main[11]),
                 $signed(shadow_main[12]), $signed(shadow_main[13]));
        $display("  mult2 mem[14..17] = %0d %0d %0d %0d  (expect 1 2 3 4)",
                 $signed(shadow_main[14]), $signed(shadow_main[15]),
                 $signed(shadow_main[16]), $signed(shadow_main[17]));
        check((shadow_main[14] === 8'd1) && (shadow_main[15] === 8'd2) &&
              (shadow_main[16] === 8'd3) && (shadow_main[17] === 8'd4),
              "second mult result = [1,2,3,4] (MACs cleared between mults)");

        // ===================================================================
        // Test 11: Signed mult (v0.2.1) — negative operands through the array.
        //   rs1 @0x000002 = [[16,0],[0,16]]        -> flat 2..5  (identity*16)
        //   rs2 @0x000006 = [[-16,32],[48,-64]]    -> flat 6..9  (mixed sign)
        //   rd  @0x00000A                           -> flat 10..13
        //   With rs1 = identity*16, C = rs2 * 16, requantized (>>>8) = rs2/16:
        //     C[0][0] = 16*(-16) = -256  >>>8 = -1
        //     C[0][1] = 16*  32  =  512  >>>8 =  2
        //     C[1][0] = 16*  48  =  768  >>>8 =  3
        //     C[1][1] = 16*(-64) = -1024 >>>8 = -4  -> [[-1,2],[3,-4]]
        //   Old unsigned MAC: -16 (0xF0) reads as +240 -> 16*240 = 3840 >>>8 =
        //   15 (positive), so mem[10] would be +15, failing this check.
        // ===================================================================
        $display("Test 11: signed mult with negative operands (v0.2.1 MAC fix)");
        rst = 0; repeat (4) @(posedge clk); rst = 1; repeat (4) @(posedge clk);
        mode_select = 1'b0;
        for (si = 0; si < 1024; si = si + 1) shadow_main[si] = 8'hEE;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_mem4(24'h000002, 8'd16, 8'd0, 8'd0, 8'd16);            // rs1 identity*16
        send_mem4(24'h000006, -8'sd16, 8'd32, 8'd48, -8'sd64);      // rs2 mixed sign
        send_program(mk_mul(24'h000002, 4'd2, 4'd2, 24'h000006, 4'd2, 4'd2, 24'h00000A));
        send_program(128'd0); // end
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        wait_feeder_done(200000);
        repeat (60) @(posedge clk);
        $display("  signed mult mem[10..13] = %0d %0d %0d %0d  (expect -1 2 3 -4)",
                 $signed(shadow_main[10]), $signed(shadow_main[11]),
                 $signed(shadow_main[12]), $signed(shadow_main[13]));
        check(($signed(shadow_main[10]) === -8'sd1) && (shadow_main[11] === 8'd2) &&
              (shadow_main[12] === 8'd3) && ($signed(shadow_main[13]) === -8'sd4),
              "signed mult with negative operands = [-1,2,3,-4]");

        // ===================================================================
        // Test 12: per-layer dyadic requantization (v0.3) — a mult carrying a
        //   non-trivial M0/n, verifying the runtime requant datapath end to end.
        //   rs1 @0x000002 = [[1,0],[0,1]]      -> flat 2..5  (identity)
        //   rs2 @0x000006 = [[100,50],[25,10]] -> flat 6..9
        //   rd  @0x00000A                       -> flat 10..13
        //   With rs1 = identity, C = rs2. Requant M0=3, n=4:
        //     result = clamp((3*x + (1<<3)) >> 4) = (3x + 8) >> 4
        //     x=100 -> (308)>>4 = 19    x=50 -> (158)>>4 = 9
        //     x=25  -> ( 83)>>4 = 5     x=10 -> ( 38)>>4 = 2   -> [[19,9],[5,2]]
        //   A truncating (no-bias) or fixed-shift requant would not match, so
        //   this fails unless the M0/n datapath is wired and rounded correctly.
        // ===================================================================
        $display("Test 12: per-layer dyadic requant (M0=3, n=4)");
        rst = 0; repeat (4) @(posedge clk); rst = 1; repeat (4) @(posedge clk);
        mode_select = 1'b0;
        for (si = 0; si < 1024; si = si + 1) shadow_main[si] = 8'hEE;
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);
        send_mem4(24'h000002, 8'd1,   8'd0,  8'd0,  8'd1);  // rs1 identity
        send_mem4(24'h000006, 8'd100, 8'd50, 8'd25, 8'd10); // rs2
        send_program(mk_mul_rq(24'h000002, 4'd2, 4'd2, 24'h000006, 4'd2, 4'd2,
                               24'h00000A, 8'd3, 8'd4));
        send_program(128'd0); // end
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        wait_feeder_done(200000);
        repeat (60) @(posedge clk);
        $display("  requant mem[10..13] = %0d %0d %0d %0d  (expect 19 9 5 2)",
                 $signed(shadow_main[10]), $signed(shadow_main[11]),
                 $signed(shadow_main[12]), $signed(shadow_main[13]));
        check((shadow_main[10] === 8'd19) && (shadow_main[11] === 8'd9) &&
              (shadow_main[12] === 8'd5)  && (shadow_main[13] === 8'd2),
              "per-layer requant (M0=3,n=4) = [19,9,5,2]");

        // ===================================================================
        // Test 13 (v0.5): tiled matmul with leading-dimension addressing.
        //   C(2x2) = A(2x16) * B(16x2), K=16 tiled into two K=8 passes:
        //     multip:  C  = A[:,0:8]  * B[0:8,:]
        //     mult:    C += A[:,8:16] * B[8:16,:]   (then requant + clear)
        //   Parents (flat, contiguous): A @0x02 (2x16, ld1=16), B @0x40 (16x2,
        //   ld2=2), C @0x80. A row0 = all 1s, A row1 = all 2s; B[k] = [1,2].
        //     C[i][j] = sum_{k=0..15} A[i][k]*B[k][j]
        //     C = [[16,32],[32,64]]; requant (M0=1,n=4) = (x+8)>>4 = [[1,2],[2,4]]
        //   The stride ld1=16 forces each 2x8 A sub-block to be read in place
        //   (row 1 begins at base+16, not base+8), and the accumulator persists
        //   from the multip into the mult (in-array tiled contraction).
        // ===================================================================
        $display("Test 13: tiled matmul (stride + multip run terminated by mult)");
        rst = 0; repeat (4) @(posedge clk); rst = 1; repeat (4) @(posedge clk);
        mode_select = 1'b0;
        for (si = 0; si < 1024; si = si + 1) shadow_main[si] = 8'hEE;

        // Single FLASH session: MEM(A), MEM(B), then the tiled-matmul program.
        spi_ss = 0;
        send_spi_byte(FLASH_CODE);

        // A parent (row-major 2x16): row0 = 1s, row1 = 2s.
        for (m = 0;  m < 16; m = m + 1) send_buf[m] = 8'd1;
        for (m = 16; m < 32; m = m + 1) send_buf[m] = 8'd2;
        send_mem_block(24'h000002, 32);   // A @ flat 2..33

        // B parent (row-major 16x2): B[k] = [1,2].
        for (m = 0; m < 16; m = m + 1) begin
            send_buf[2*m]     = 8'd1;
            send_buf[2*m + 1] = 8'd2;
        end
        send_mem_block(24'h000040, 32);   // B @ flat 64..95

        // Program: stride, multip (tile 0), mult (tile 1, requant M0=1 n=4), end.
        send_program(mk_stride(24'd16, 24'd2));
        send_program(mk_multip(24'h000002, 4'd2, 4'd8, 24'h000040, 4'd8, 4'd2,
                               24'h000080, 8'd1, 8'd0));
        send_program(mk_mul_rq(24'h00000A, 4'd2, 4'd8, 24'h000050, 4'd8, 4'd2,
                               24'h000080, 8'd1, 8'd4));
        send_program(128'd0); // end
        send_spi_byte(STOP_CODE);
        spi_ss = 1;
        wait_feeder_done(400000);
        repeat (120) @(posedge clk);
        $display("  tiled C mem[128..131] = %0d %0d %0d %0d  (expect 1 2 2 4)",
                 $signed(shadow_main[128]), $signed(shadow_main[129]),
                 $signed(shadow_main[130]), $signed(shadow_main[131]));
        check((shadow_main[128] === 8'd1) && (shadow_main[129] === 8'd2) &&
              (shadow_main[130] === 8'd2) && (shadow_main[131] === 8'd4),
              "tiled matmul (multip+mult, ld1=16) = [[1,2],[2,4]]");

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
