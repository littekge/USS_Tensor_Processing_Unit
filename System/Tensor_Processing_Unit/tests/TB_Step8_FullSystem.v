/*
 * File: TB_Step8_FullSystem.v
 * Date: 2026-06-22
 *
 * Testbench for Build Step 8: full-system integration test of the TPU.
 *
 * This testbench drives the top-level TPU module exactly as the real hardware
 * would be driven: a single SPI session (per the Functional TPU Message
 * Protocol) programs test weights and a program that exercises every ISA
 * instruction (mult, relu, add, syscall). After the STOP code the Programmer
 * releases the TPU (program HIGH -> trst HIGH) and the Feeder/Controller run
 * the program to completion. Results are checked by snooping the (muxed)
 * weight-memory and 0x1-buffer write ports into shadow memories, which mirror
 * exactly what the real RAMs store. No internal RAM array is referenced.
 *
 * Unified address space: a MEM-packet address and an ISA instruction address
 * refer to the SAME logical location (the ISA reserves 0x0 and 0x1; weight
 * memory begins at ISA 0x2). The physical weight-memory address for ISA
 * address A (A >= 2) is A - 2.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. Open a terminal and cd to the project root:
 *        cd <path>/Tensor_Processing_Unit
 *
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
 *
 *   3. Simulate (do NOT optimize signals away; log all waveforms):
 *        vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step8_FullSystem -do "add wave -r /*; run -all"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary prints total pass/fail counts.
 *   All 7 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: Programming completes — program returns HIGH after STOP and the
 *           programmed input weights land in weight memory (SPI -> Programmer
 *           -> Weight_Memory path).
 *   Test 2: syscall terminates fetch — the Feeder reaches its DONE state.
 *   Test 3: mult — C = rs1 * rs2 written row-major and requantized correctly.
 *   Test 4: relu — max(0, x) applied element-wise and written to dest.
 *   Test 5: add  — element-wise sum written to dest.
 *   Test 6: add saturation — operands that overflow int8 clamp to -128 / +127.
 *   Test 7: clean completion — Controller returns to IDLE with no error state
 *           and the Feeder is in DONE (whole program executed without faults).
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

    // -----------------------------------------------------------------------
    // Timing parameters
    // -----------------------------------------------------------------------
    localparam CLK_HALF = 10;   // 10 ns => 50 MHz FPGA clock
    localparam SPI_HALF = 100;  // 100 ns => 5 MHz SPI clock

    // Protocol function codes
    localparam [7:0] START_CODE = 8'h55;
    localparam [7:0] STOP_CODE  = 8'h53;
    localparam [7:0] MEM_CODE   = 8'h4D;
    localparam [7:0] PROG_CODE  = 8'h50;

    // FSM state encodings to observe (must match the DUT modules)
    localparam [3:0] FEEDER_DONE = 4'd7;   // Feeder.v DONE
    localparam [3:0] CTRL_IDLE   = 4'd2;   // Controller.v IDLE

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    TPU dut (
        .i_clk       (clk),
        .i_rst       (rst),
        .i_SPI_Clk   (spi_clk),
        .o_SPI_MISO  (spi_miso),
        .i_SPI_MOSI  (spi_mosi),
        .i_SPI_SS    (spi_ss),
        .o_debug_val (debug_val)
    );

    // -----------------------------------------------------------------------
    // Clock generation
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #CLK_HALF clk = ~clk;

    // -----------------------------------------------------------------------
    // Shadow memories — mirror the muxed write ports of the device RAMs.
    // These reproduce exactly what the real RAMs store (write on posedge when
    // wren HIGH) without referencing any vendor-IP internal array.
    // -----------------------------------------------------------------------
    reg [7:0] shadow_wm  [0:255];
    reg [7:0] shadow_buf [0:255];

    always @(posedge clk) begin
        if (dut.wm_wren_a)  shadow_wm [dut.wm_address_a[7:0]]  <= dut.wm_data_a;
        if (dut.buf_wren_a) shadow_buf[dut.buf_address_a[7:0]] <= dut.buf_data_a;
    end

    // -----------------------------------------------------------------------
    // Test bookkeeping
    // -----------------------------------------------------------------------
    integer pass_count;
    integer fail_count;

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

    // -----------------------------------------------------------------------
    // SPI byte transmission (mode 0, MSB first). SS is held LOW for the whole
    // session; bytes are streamed back-to-back.
    // -----------------------------------------------------------------------
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

    // Send a 4-element MEM block to a 16-bit address.
    task send_mem4;
        input [15:0] addr;
        input [7:0]  b0, b1, b2, b3;
        begin
            send_spi_byte(MEM_CODE);
            send_spi_byte(addr[7:0]);
            send_spi_byte(addr[15:8]);
            send_spi_byte(8'd4);   // LLEN = 4
            send_spi_byte(8'd0);   // ULEN = 0
            send_spi_byte(b0);
            send_spi_byte(b1);
            send_spi_byte(b2);
            send_spi_byte(b3);
        end
    endtask

    // Send a 128-bit instruction as a PROGRAM packet (LSB byte first).
    task send_program;
        input [127:0] instr;
        integer i;
        begin
            send_spi_byte(PROG_CODE);
            for (i = 0; i < 16; i = i + 1)
                send_spi_byte(instr[i*8 +: 8]);
        end
    endtask

    // -----------------------------------------------------------------------
    // ISA instruction builders (see Functional_TPU_ISA_v0.1.md)
    // -----------------------------------------------------------------------
    function [127:0] mk_mul;
        input [15:0] rs1;
        input [3:0]  sz11, sz12;
        input [15:0] rs2;
        input [3:0]  sz21, sz22;
        input [15:0] rd;
        begin
            mk_mul = {4'b1000, rs1, sz11, sz12, rs2, sz21, sz22, rd, 57'd0, 3'd0};
        end
    endfunction

    function [127:0] mk_relu;
        input [15:0] rs1;
        input [15:0] rd;
        input [15:0] len;
        begin
            mk_relu = {4'b1001, rs1, rd, len, 73'd0, 3'd0};
        end
    endfunction

    function [127:0] mk_add;
        input [15:0] rs1;
        input [15:0] rs2;
        input [7:0]  sz1, sz2;
        input [15:0] rd;
        begin
            mk_add = {4'b1010, rs1, rs2, sz1, sz2, rd, 57'd0, 3'd0};
        end
    endfunction

    // -----------------------------------------------------------------------
    // Wait helper: poll the Feeder until it reaches DONE (syscall fetched).
    // Returns the number of cycles waited; caller checks for timeout.
    // -----------------------------------------------------------------------
    integer wait_cycles;
    task wait_feeder_done;
        input integer max_cycles;
        begin
            wait_cycles = 0;
            while ((dut.feeder.S !== FEEDER_DONE) && (wait_cycles < max_cycles)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
            end
        end
    endtask

    // -----------------------------------------------------------------------
    // Main test sequence
    // -----------------------------------------------------------------------
    integer i;

    initial begin
        // Init
        rst        = 0;
        spi_clk    = 0;
        spi_mosi   = 0;
        spi_ss     = 1;
        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < 256; i = i + 1) begin
            shadow_wm[i]  = 8'd0;
            shadow_buf[i] = 8'd0;
        end

        // Apply and release global reset
        repeat (4) @(posedge clk);
        rst = 1;
        repeat (4) @(posedge clk);

        // -------------------------------------------------------------------
        // Single SPI programming session.
        //
        // Memory map (ISA address -> physical weight-memory address = ISA-2):
        //   MULT  rs1 @0x02 = [[16,0],[0,16]]       (2x2)
        //         rs2 @0x06 = [[16,32],[48,64]]     (2x2)
        //         rd  @0x0A  expect [[1,2],[3,4]]    (256/512/768/1024 >> 8)
        //   RELU  rs1 @0x0E = [-5, 7, -1, 100]      (len 4)
        //         rd  @0x12  expect [0, 7, 0, 100]
        //   ADD   rs1 @0x16 = [10, 20, -100, 50]    (2x2)
        //         rs2 @0x1A = [5, -3, -50, 80]      (2x2)
        //         rd  @0x1E  expect [15, 17, -128, 127] (saturating)
        // -------------------------------------------------------------------
        spi_ss = 0;
        send_spi_byte(START_CODE);

        // Input weights / activations
        send_mem4(16'h0002, 8'd16, 8'd0,  8'd0,  8'd16);  // mult rs1
        send_mem4(16'h0006, 8'd16, 8'd32, 8'd48, 8'd64);  // mult rs2
        send_mem4(16'h000E, 8'hFB, 8'h07, 8'hFF, 8'h64);  // relu src (-5,7,-1,100)
        send_mem4(16'h0016, 8'h0A, 8'h14, 8'h9C, 8'h32);  // add  rs1 (10,20,-100,50)
        send_mem4(16'h001A, 8'h05, 8'hFD, 8'hCE, 8'h50);  // add  rs2 (5,-3,-50,80)

        // Program: mult, relu, add, syscall
        send_program(mk_mul(16'h0002, 4'd2, 4'd2, 16'h0006, 4'd2, 4'd2, 16'h000A));
        send_program(mk_relu(16'h000E, 16'h0012, 16'd4));
        send_program(mk_add(16'h0016, 16'h001A, 8'd2, 8'd2, 16'h001E));
        send_program(128'd0);   // syscall (opcode 0000)

        send_spi_byte(STOP_CODE);
        spi_ss = 1;

        // -------------------------------------------------------------------
        // Run the program to completion (Feeder reaches DONE at the syscall).
        // -------------------------------------------------------------------
        wait_feeder_done(200000);
        repeat (50) @(posedge clk);   // margin for final writeback to settle

        // -------------------------------------------------------------------
        // Diagnostics
        // -------------------------------------------------------------------
        $display("");
        $display("feeder.S = %0d (DONE=%0d), ctrl.S = %0d (IDLE=%0d), program=%b, wait=%0d",
                 dut.feeder.S, FEEDER_DONE, dut.ctrl.S, CTRL_IDLE, dut.program, wait_cycles);
        $display("mult inputs  WM[0..3]  = %0d %0d %0d %0d",
                 $signed(shadow_wm[0]), $signed(shadow_wm[1]), $signed(shadow_wm[2]), $signed(shadow_wm[3]));
        $display("mult inputs  WM[4..7]  = %0d %0d %0d %0d",
                 $signed(shadow_wm[4]), $signed(shadow_wm[5]), $signed(shadow_wm[6]), $signed(shadow_wm[7]));
        $display("mult result  WM[8..11] = %0d %0d %0d %0d  (expect 1 2 3 4)",
                 $signed(shadow_wm[8]), $signed(shadow_wm[9]), $signed(shadow_wm[10]), $signed(shadow_wm[11]));
        $display("relu result  WM[16..19]= %0d %0d %0d %0d  (expect 0 7 0 100)",
                 $signed(shadow_wm[16]), $signed(shadow_wm[17]), $signed(shadow_wm[18]), $signed(shadow_wm[19]));
        $display("add  result  WM[28..31]= %0d %0d %0d %0d  (expect 15 17 -128 127)",
                 $signed(shadow_wm[28]), $signed(shadow_wm[29]), $signed(shadow_wm[30]), $signed(shadow_wm[31]));
        $display("");

        // -------------------------------------------------------------------
        // Test 1: programming completed and inputs landed in memory
        // -------------------------------------------------------------------
        $display("Test 1: programming over SPI completes and weights stored");
        check((dut.program === 1'b1) &&
              (shadow_wm[0] === 8'd16) && (shadow_wm[1] === 8'd0) &&
              (shadow_wm[2] === 8'd0)  && (shadow_wm[3] === 8'd16) &&
              (shadow_wm[4] === 8'd16) && (shadow_wm[7] === 8'd64),
              "program HIGH after STOP and mult inputs in weight memory");

        // -------------------------------------------------------------------
        // Test 2: syscall stops the Feeder
        // -------------------------------------------------------------------
        $display("Test 2: syscall terminates instruction fetch");
        check((dut.feeder.S === FEEDER_DONE) && (wait_cycles < 200000),
              "Feeder reached DONE after syscall");

        // -------------------------------------------------------------------
        // Test 3: mult result
        // -------------------------------------------------------------------
        $display("Test 3: mult result C = rs1 * rs2 (requantized)");
        check((shadow_wm[8]  === 8'd1) && (shadow_wm[9]  === 8'd2) &&
              (shadow_wm[10] === 8'd3) && (shadow_wm[11] === 8'd4),
              "mult result [1,2,3,4]");

        // -------------------------------------------------------------------
        // Test 4: relu result
        // -------------------------------------------------------------------
        $display("Test 4: relu result max(0,x)");
        check((shadow_wm[16] === 8'd0) && (shadow_wm[17] === 8'd7) &&
              (shadow_wm[18] === 8'd0) && (shadow_wm[19] === 8'd100),
              "relu result [0,7,0,100]");

        // -------------------------------------------------------------------
        // Test 5: add result
        // -------------------------------------------------------------------
        $display("Test 5: add result element-wise sum");
        check((shadow_wm[28] === 8'd15) && (shadow_wm[29] === 8'd17),
              "add result first two elements [15,17]");

        // -------------------------------------------------------------------
        // Test 6: add saturation (-150 -> -128, +130 -> +127)
        // -------------------------------------------------------------------
        $display("Test 6: add saturation clamps to int8 range");
        check((shadow_wm[30] === 8'h80) && (shadow_wm[31] === 8'h7F),
              "add saturates [-128, 127]");

        // -------------------------------------------------------------------
        // Test 7: clean completion (no error states)
        // -------------------------------------------------------------------
        $display("Test 7: program ran to completion with no fault states");
        check((dut.ctrl.S === CTRL_IDLE) && (dut.feeder.S === FEEDER_DONE),
              "Controller IDLE and Feeder DONE at end");

        // -------------------------------------------------------------------
        // Summary
        // -------------------------------------------------------------------
        $display("");
        $display("================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("================================");
        $finish;
    end

endmodule
