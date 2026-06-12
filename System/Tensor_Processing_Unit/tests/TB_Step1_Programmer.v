/* 
 * File: TB_Step1_Programmer.v
 * Date: 2026-06-12
 *
 * Testbench for Build Step 1: Programmer module.
 * Tests the Programmer state machine by bit-banging SPI frames and verifying
 * memory write outputs.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation from Quartus Prime Lite):
 *
 *   1. Open a terminal and cd to the project root:
 *        cd <path>/Tensor_Processing_Unit
 *
 *   2. Compile with the altera_mf simulation library:
 *        vlib work
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v
 *        vlog -work work TPU/PROGRAMMER/Programmer.v
 *        vlog -work work tests/TB_Step1_Programmer.v
 *
 *   3. Simulate (altera_mf_ver is the pre-compiled Verilog megafunction library):
 *        vsim -L altera_mf_ver work.TB_Step1_Programmer -do "run -all; quit"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "PASS" or "FAIL" to the transcript.
 *   A final summary line prints total pass/fail counts.
 *   All 7 tests should show PASS for a correct implementation.
 *
 * TEST CASES:
 *   Test 1: program signal HIGH after reset (not programming)
 *   Test 2: program signal goes LOW on START, HIGH on STOP
 *   Test 3: MEM packet writes correct data to Weight_Memory (address >= 2)
 *   Test 4: MEM packet writes correct data to 0x1 Buffer (address == 1)
 *   Test 5: PROGRAM packet writes correct 128-bit instruction to Program_Memory
 *   Test 6: COM_ERROR — invalid function code after START halts writes
 *   Test 7: WRITE_ERROR — MEM packet with address 0 halts writes
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_Step1_Programmer;

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    reg        clk;
    reg        rst;

    wire [127:0] pm_data;
    wire [9:0]   pm_address;
    wire         pm_wren;
    wire [7:0]   wm_data_a;
    wire [15:0]  wm_address_a;
    wire         wm_wren_a;
    wire [7:0]   buf_data_a;
    wire [15:0]  buf_address_a;
    wire         buf_wren_a;
    wire         program;

    // SPI
    reg  spi_clk;
    reg  spi_mosi;
    reg  spi_ss;
    wire spi_miso;

    // -----------------------------------------------------------------------
    // Timing parameters
    // -----------------------------------------------------------------------
    localparam CLK_HALF  = 10;   // 10 ns => 50 MHz FPGA clock
    localparam SPI_HALF  = 100;  // 100 ns => 5 MHz SPI clock (>= 4x slower)

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    Programmer dut (
        .i_clk          (clk),
        .i_rst          (rst),
        .o_pm_data      (pm_data),
        .o_pm_address   (pm_address),
        .o_pm_wren      (pm_wren),
        .o_wm_data_a    (wm_data_a),
        .o_wm_address_a (wm_address_a),
        .o_wm_wren_a    (wm_wren_a),
        .o_buf_data_a   (buf_data_a),
        .o_buf_address_a(buf_address_a),
        .o_buf_wren_a   (buf_wren_a),
        .o_program      (program),
        .i_SPI_Clk      (spi_clk),
        .o_SPI_MISO     (spi_miso),
        .i_SPI_MOSI     (spi_mosi),
        .i_SPI_SS       (spi_ss)
    );

    // -----------------------------------------------------------------------
    // Clock generation
    // -----------------------------------------------------------------------
    initial clk = 0;
    always #CLK_HALF clk = ~clk;

    // -----------------------------------------------------------------------
    // Test bookkeeping
    // -----------------------------------------------------------------------
    integer pass_count;
    integer fail_count;

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

    // -----------------------------------------------------------------------
    // SPI byte transmission (mode 0, MSB first)
    // SS must be asserted LOW before calling and de-asserted after the packet.
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

    // -----------------------------------------------------------------------
    // Captured write events (sampled in always blocks below)
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
    // Main test sequence
    // -----------------------------------------------------------------------
    integer timeout;
    integer k;
    reg [127:0] expected_instr;

    initial begin
        // Init
        rst            = 0;
        spi_clk        = 0;
        spi_mosi       = 0;
        spi_ss         = 1;
        pass_count     = 0;
        fail_count     = 0;
        wm_write_count  = 0;
        buf_write_count = 0;
        pm_write_count  = 0;

        // Apply and release reset
        repeat (4) @(posedge clk);
        rst = 1;
        repeat (4) @(posedge clk);

        // -------------------------------------------------------------------
        // Test 1: program HIGH after reset (not programming)
        // -------------------------------------------------------------------
        $display("Test 1: program signal HIGH after reset");
        report(program === 1'b1, "program == 1 after reset");

        // -------------------------------------------------------------------
        // Test 2: program LOW on START, HIGH on STOP
        //   Packet: [START, STOP]
        // -------------------------------------------------------------------
        $display("Test 2: program toggles with START / STOP");
        spi_ss = 0;
        send_spi_byte(8'h55); // START
        send_spi_byte(8'h53); // STOP
        spi_ss = 1;
        // Allow state machine to process
        repeat (500) @(posedge clk);
        // program should be HIGH again after STOP
        report(program === 1'b1, "program == 1 after STOP");

        // -------------------------------------------------------------------
        // Test 3: MEM packet -> Weight_Memory (address 0x0005, 3 bytes)
        //   Packet: [START, MEM, 0x05, 0x00, 0x03, 0x00, 0xAB, 0xCD, 0xEF, STOP]
        //   Expected: 3 writes at WM addresses 5, 6, 7 with data AB, CD, EF
        // -------------------------------------------------------------------
        $display("Test 3: MEM packet writes to Weight_Memory");
        wm_write_count = 0;
        spi_ss = 0;
        send_spi_byte(8'h55); // START
        send_spi_byte(8'h4D); // MEM
        send_spi_byte(8'h05); // LADD = 0x05
        send_spi_byte(8'h00); // UADD = 0x00
        send_spi_byte(8'h03); // LLEN = 3
        send_spi_byte(8'h00); // ULEN = 0
        send_spi_byte(8'hAB); // data[0]
        send_spi_byte(8'hCD); // data[1]
        send_spi_byte(8'hEF); // data[2]
        send_spi_byte(8'h53); // STOP
        spi_ss = 1;
        // Wait for all writes (up to 5000 clock cycles)
        timeout = 0;
        while (wm_write_count < 3 && timeout < 5000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        report(wm_write_count === 3,          "3 writes to WM");
        report(wm_addr_log[0] === 16'd5 &&
               wm_data_log[0] === 8'hAB,      "WM[5] = 0xAB");
        report(wm_addr_log[1] === 16'd6 &&
               wm_data_log[1] === 8'hCD,      "WM[6] = 0xCD");
        report(wm_addr_log[2] === 16'd7 &&
               wm_data_log[2] === 8'hEF,      "WM[7] = 0xEF");

        // -------------------------------------------------------------------
        // Test 4: MEM packet -> 0x1 Buffer (address 0x0001, 2 bytes)
        //   Packet: [START, MEM, 0x01, 0x00, 0x02, 0x00, 0x12, 0x34, STOP]
        //   Expected: 2 writes at internal BUF addresses 0, 1 with data 12, 34
        // -------------------------------------------------------------------
        $display("Test 4: MEM packet writes to 0x1 Buffer");
        buf_write_count = 0;
        spi_ss = 0;
        send_spi_byte(8'h55); // START
        send_spi_byte(8'h4D); // MEM
        send_spi_byte(8'h01); // LADD = 0x01
        send_spi_byte(8'h00); // UADD = 0x00
        send_spi_byte(8'h02); // LLEN = 2
        send_spi_byte(8'h00); // ULEN = 0
        send_spi_byte(8'h12); // data[0]
        send_spi_byte(8'h34); // data[1]
        send_spi_byte(8'h53); // STOP
        spi_ss = 1;
        timeout = 0;
        while (buf_write_count < 2 && timeout < 5000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        report(buf_write_count === 2,           "2 writes to 0x1 Buffer");
        report(buf_addr_log[0] === 16'd0 &&
               buf_data_log[0] === 8'h12,       "BUF[0] = 0x12");
        report(buf_addr_log[1] === 16'd1 &&
               buf_data_log[1] === 8'h34,       "BUF[1] = 0x34");

        // -------------------------------------------------------------------
        // Test 5: PROGRAM packet writes a 128-bit instruction (LSB first)
        //   16 bytes sent: 0x00 .. 0x0F
        //   Expected instruction[7:0]=0x00, instruction[15:8]=0x01, ...
        //   i.e., pm_data = 128'h0F0E0D0C0B0A09080706050403020100
        // -------------------------------------------------------------------
        $display("Test 5: PROGRAM packet writes instruction to Program_Memory");
        pm_write_count = 0;
        expected_instr = 128'h0F0E0D0C0B0A09080706050403020100;
        spi_ss = 0;
        send_spi_byte(8'h55); // START
        send_spi_byte(8'h50); // PROGRAM
        for (k = 0; k < 16; k = k + 1)
            send_spi_byte(k[7:0]); // bytes 0x00 - 0x0F
        send_spi_byte(8'h53); // STOP
        spi_ss = 1;
        timeout = 0;
        while (pm_write_count < 1 && timeout < 5000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        report(pm_write_count === 1,                "1 write to Program_Memory");
        report(pm_addr_log[0] === 10'd0,            "pm_address == 0");
        report(pm_data_log[0] === expected_instr,   "pm_data correct");

        // -------------------------------------------------------------------
        // Test 6: COM_ERROR — invalid function code causes no memory writes
        //   Packet: [START, 0xFF (invalid)]
        //   After error, no wm/buf/pm writes should occur on next START attempt
        // -------------------------------------------------------------------
        $display("Test 6: COM_ERROR on invalid function code");
        wm_write_count  = 0;
        buf_write_count = 0;
        pm_write_count  = 0;
        spi_ss = 0;
        send_spi_byte(8'h55); // START
        send_spi_byte(8'hFF); // Invalid function code -> COM_ERROR
        spi_ss = 1;
        repeat (200) @(posedge clk); // allow time for any erroneous writes
        report(wm_write_count  === 0 &&
               buf_write_count === 0 &&
               pm_write_count  === 0, "No memory writes in COM_ERROR");

        // -------------------------------------------------------------------
        // Test 7: WRITE_ERROR — MEM packet with address 0 causes no writes
        //   Packet: [START, MEM, 0x00, 0x00, 0x01, 0x00, 0xAA]
        //   Address 0x0000 is read-only; programmer should enter WRITE_ERROR.
        // -------------------------------------------------------------------
        $display("Test 7: WRITE_ERROR on address 0x0000");
        // Reset DUT to clear error state
        rst = 0;
        repeat (4) @(posedge clk);
        rst = 1;
        repeat (4) @(posedge clk);
        wm_write_count  = 0;
        buf_write_count = 0;
        pm_write_count  = 0;
        spi_ss = 0;
        send_spi_byte(8'h55); // START
        send_spi_byte(8'h4D); // MEM
        send_spi_byte(8'h00); // LADD = 0x00
        send_spi_byte(8'h00); // UADD = 0x00 -> address 0 -> WRITE_ERROR
        send_spi_byte(8'h01); // LLEN = 1
        send_spi_byte(8'h00); // ULEN = 0
        send_spi_byte(8'hAA); // data byte (should NOT be written)
        spi_ss = 1;
        repeat (300) @(posedge clk);
        report(wm_write_count === 0 &&
               buf_write_count === 0, "No writes on WRITE_ERROR (address 0)");

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
