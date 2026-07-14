/*
 * File: TB_SPI_Slave.v
 * Date: 2026-07-14
 *
 * Dedicated unit testbench for the v0.5.1 SPI_Slave synchronous-oversampling
 * rewrite (TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v). The module ports, the
 * SPI_MODE parameter, and the functional contract are UNCHANGED from the prior
 * (nandland) version; this bench proves both functional equivalence and the
 * robustness the old raw-SCK/raw-CS design lacked.
 *
 * Four DUTs (SPI_MODE 0..3) share one set of SPI stimulus nets (sck/mosi/cs).
 * Only the DUT whose mode matches the driven clocking is checked for a given
 * test; the shared nets keep the bench compact while still exercising every
 * CPOL/CPHA combination. Per-DUT monitors record the received byte, the number
 * of o_RX_DV pulses (ev), and the total number of i_Clk cycles o_RX_DV was
 * high (hi). For a correct single-cycle pulse, hi == ev.
 *
 * -------------------------------------------------------------------------
 * HOW TO RUN (Questa Intel FPGA simulation):
 *
 *   1. cd <path>/Tensor_Processing_Unit
 *   2. Compile:
 *        vlib work
 *        vlog -work work TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v
 *        vlog -work work tests/TB_SPI_Slave.v
 *   3. Simulate (log every signal; do NOT optimize signals away):
 *        vsim -voptargs=+acc work.TB_SPI_Slave -do "add wave -r /*; run -all; quit"
 *
 * -------------------------------------------------------------------------
 * PASS / FAIL CRITERIA:
 *   Each test prints "Test N: PASS - ..." or "Test N: FAIL - ...". A final
 *   "Results: X PASS, Y FAIL" line summarizes. A correct implementation prints
 *   ZERO "Test N: FAIL" lines (this exact substring is what the regression
 *   runner greps for).
 *
 * TEST CASES:
 *   Test 1  : MSB-first assembly of a single byte (0xA5); DV is one i_Clk cycle.
 *   Test 2  : A second, different byte (0x3C) received correctly.
 *   Test 3  : Multiple bytes in ONE CS-low window (0x11,0x22,0x33) in order.
 *   Test 4  : Back-to-back bytes with no inter-byte idle (0xF0 then 0x0F).
 *   Test 5  : A short SCK glitch mid-byte is rejected (byte + DV unaffected).
 *   Test 6  : A short CS-deassert glitch mid-frame is rejected (frame intact).
 *   Test 7  : Slow/noisy (bouncing) SCK edges still yield the correct byte.
 *   Test 8  : CS deasserted right after the final SCK edge still delivers the
 *             last byte (the old design's last-byte/CS race).
 *   Test 9  : Over a multi-byte burst, every DV pulse is exactly one i_Clk
 *             cycle (hi == ev).
 *   Test 10 : Active-low reset clears o_RX_DV and o_RX_Byte.
 *   Test 11 : TX/MISO: tri-state while CS high; MSB preload on CS-assert;
 *             MSB-first shift; i_TX_Byte registered on i_TX_DV.
 *   Test 12 : SPI_MODE 1 (CPOL0/CPHA1) samples on the trailing edge (0x93).
 *   Test 13 : SPI_MODE 2 (CPOL1/CPHA0) samples on the leading edge (0x5A).
 *   Test 14 : SPI_MODE 3 (CPOL1/CPHA1) samples on the trailing edge (0xC7).
 * -------------------------------------------------------------------------
 */

`timescale 1 ns / 1 ps

module TB_SPI_Slave;

    // -----------------------------------------------------------------------
    // Timing
    // -----------------------------------------------------------------------
    localparam CLK_HALF  = 10;    // 50 MHz i_Clk (20 ns period)
    localparam SPI_HALF  = 200;   // 2.5 MHz SPI half-period (10 i_Clk samples)
    localparam GLITCH_W  = 30;    // < DEBOUNCE_N i_Clk window -> must be rejected

    // -----------------------------------------------------------------------
    // DUT / stimulus signals
    // -----------------------------------------------------------------------
    reg        clk;
    reg        rst;       // i_Rst_L, active low

    // Shared SPI stimulus (fanned out to all four DUTs).
    reg        sck;
    reg        mosi;
    reg        cs;        // i_SPI_CS_n (active low)

    // TX stimulus (DUT0 only; other DUTs tie TX off like the in-system wiring).
    reg        tx_dv;
    reg  [7:0] tx_byte;

    wire [3:0] dv;                 // o_RX_DV of each DUT
    wire [7:0] rx0, rx1, rx2, rx3; // o_RX_Byte of each DUT
    wire       miso0, miso1, miso2, miso3;

    // -----------------------------------------------------------------------
    // DUTs: SPI_MODE 0..3
    // -----------------------------------------------------------------------
    SPI_Slave #(.SPI_MODE(0)) dut0 (
        .i_Rst_L(rst), .i_Clk(clk),
        .o_RX_DV(dv[0]), .o_RX_Byte(rx0),
        .i_TX_DV(tx_dv), .i_TX_Byte(tx_byte),
        .i_SPI_Clk(sck), .o_SPI_MISO(miso0), .i_SPI_MOSI(mosi), .i_SPI_CS_n(cs)
    );

    SPI_Slave #(.SPI_MODE(1)) dut1 (
        .i_Rst_L(rst), .i_Clk(clk),
        .o_RX_DV(dv[1]), .o_RX_Byte(rx1),
        .i_TX_DV(1'b0), .i_TX_Byte(8'h00),
        .i_SPI_Clk(sck), .o_SPI_MISO(miso1), .i_SPI_MOSI(mosi), .i_SPI_CS_n(cs)
    );

    SPI_Slave #(.SPI_MODE(2)) dut2 (
        .i_Rst_L(rst), .i_Clk(clk),
        .o_RX_DV(dv[2]), .o_RX_Byte(rx2),
        .i_TX_DV(1'b0), .i_TX_Byte(8'h00),
        .i_SPI_Clk(sck), .o_SPI_MISO(miso2), .i_SPI_MOSI(mosi), .i_SPI_CS_n(cs)
    );

    SPI_Slave #(.SPI_MODE(3)) dut3 (
        .i_Rst_L(rst), .i_Clk(clk),
        .o_RX_DV(dv[3]), .o_RX_Byte(rx3),
        .i_TX_DV(1'b0), .i_TX_Byte(8'h00),
        .i_SPI_Clk(sck), .o_SPI_MISO(miso3), .i_SPI_MOSI(mosi), .i_SPI_CS_n(cs)
    );

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    always #CLK_HALF clk = ~clk;

    // -----------------------------------------------------------------------
    // Per-DUT monitors
    //   cap[k] = last byte, ev[k] = #DV pulses, hi[k] = total DV-high cycles.
    //   rxlog0 records every byte DUT0 receives (order check for multi-byte).
    // -----------------------------------------------------------------------
    reg  [7:0] rxb [0:3];
    reg  [7:0] cap [0:3];
    integer    ev  [0:3];
    integer    hi  [0:3];
    reg        dvd [0:3];

    reg  [7:0] rxlog0 [0:31];
    integer    n0;

    always @(*) begin
        rxb[0] = rx0; rxb[1] = rx1; rxb[2] = rx2; rxb[3] = rx3;
    end

    // Index 0 (with per-byte logging).
    always @(posedge clk) begin
        if (dv[0]) begin
            cap[0]      = rx0;
            hi[0]       = hi[0] + 1;
            rxlog0[n0]  = rx0;
            n0          = n0 + 1;
        end
        if (dv[0] & ~dvd[0]) ev[0] = ev[0] + 1;
        dvd[0] <= dv[0];
    end

    // Indices 1..3.
    genvar gi;
    generate
        for (gi = 1; gi < 4; gi = gi + 1) begin : mon
            always @(posedge clk) begin
                if (dv[gi]) begin
                    cap[gi] = rxb[gi];
                    hi[gi]  = hi[gi] + 1;
                end
                if (dv[gi] & ~dvd[gi]) ev[gi] = ev[gi] + 1;
                dvd[gi] <= dv[gi];
            end
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Bookkeeping
    // -----------------------------------------------------------------------
    integer pass_count, fail_count;
    integer k;

    task check;
        input integer tnum;
        input pass;
        input [255:0] desc;
        begin
            if (pass) begin
                $display("Test %0d: PASS - %0s", tnum, desc);
                pass_count = pass_count + 1;
            end else begin
                $display("Test %0d: FAIL - %0s", tnum, desc);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task clear_mon;
        input integer idx;
        begin
            ev[idx]  = 0;
            hi[idx]  = 0;
            cap[idx] = 8'h00;
        end
    endtask

    // -----------------------------------------------------------------------
    // Stimulus helpers
    // -----------------------------------------------------------------------

    // Generic mode-aware byte send on the shared nets. Assumes sck already at
    // the mode's idle level (c_pol) and CS asserted; leaves sck at idle.
    task spi_send;
        input        c_pol;
        input        c_pha;
        input  [7:0] b;
        integer i;
        begin
            if (c_pha == 1'b0) begin
                // CPHA0: data set while idle, sampled on the leading edge.
                for (i = 7; i >= 0; i = i - 1) begin
                    mosi = b[i];
                    sck  = c_pol;      // idle
                    #SPI_HALF;
                    sck  = ~c_pol;     // leading edge -> slave samples
                    #SPI_HALF;
                    sck  = c_pol;      // trailing edge -> back to idle
                end
            end else begin
                // CPHA1: data changed on the leading edge, sampled on trailing.
                for (i = 7; i >= 0; i = i - 1) begin
                    sck  = ~c_pol;     // leading edge
                    mosi = b[i];
                    #SPI_HALF;
                    sck  = c_pol;      // trailing edge -> slave samples
                    #SPI_HALF;
                end
            end
        end
    endtask

    // Mode-0 single clean bit (idle low, sample on rising).
    task m0_bit;
        input v;
        begin
            mosi = v;
            sck  = 1'b0; #SPI_HALF;
            sck  = 1'b1; #SPI_HALF;
            sck  = 1'b0;
        end
    endtask

    // Mode-0 bit whose SCK edges bounce before settling (slow/noisy edge). The
    // bounces are far shorter than the debounce window, so the DUT must accept
    // exactly one rising and one falling edge -> exactly one sampled bit.
    task m0_bit_noisy;
        input v;
        begin
            mosi = v;
            sck  = 1'b0; #SPI_HALF;
            // bouncy rising edge
            sck = 1'b1; #CLK_HALF; sck = 1'b0; #CLK_HALF;
            sck = 1'b1; #CLK_HALF; sck = 1'b0; #CLK_HALF;
            sck = 1'b1; #SPI_HALF;         // settle high -> ONE accepted rise
            // bouncy falling edge
            sck = 1'b0; #CLK_HALF; sck = 1'b1; #CLK_HALF;
            sck = 1'b0; #SPI_HALF;         // settle low  -> ONE accepted fall
        end
    endtask

    // Short SCK high glitch during mode-0 idle. Must be rejected (no sample).
    task sck_glitch;
        begin
            sck = 1'b1; #GLITCH_W;
            sck = 1'b0; #SPI_HALF;
        end
    endtask

    // Short CS-deassert glitch during a frame. Must be rejected (frame intact).
    task cs_glitch;
        begin
            cs = 1'b1; #GLITCH_W;
            cs = 1'b0; #SPI_HALF;
        end
    endtask

    // Run a full mode-m byte in its own clean CS window against DUT m; returns
    // via the shared monitors. Pulses reset first so every DUT starts clean and
    // SCK begins at the mode's idle level (no manufactured first edge).
    task mode_smoke;
        input integer idx;
        input        c_pol;
        input        c_pha;
        input  [7:0] b;
        begin
            // Reset all DUTs.
            rst = 1'b0; sck = c_pol; mosi = 1'b0; cs = 1'b1;
            repeat (4) @(posedge clk);
            rst = 1'b1;
            repeat (4) @(posedge clk);

            clear_mon(idx);
            cs = 1'b0; #SPI_HALF;      // frame start
            spi_send(c_pol, c_pha, b);
            #(SPI_HALF * 4);
            cs = 1'b1; #(SPI_HALF * 2);
        end
    endtask

    // -----------------------------------------------------------------------
    // Test sequence
    // -----------------------------------------------------------------------
    initial begin
        clk        = 1'b0;
        rst        = 1'b0;
        sck        = 1'b0;
        mosi       = 1'b0;
        cs         = 1'b1;
        tx_dv      = 1'b0;
        tx_byte    = 8'h00;
        pass_count = 0;
        fail_count = 0;
        n0         = 0;
        for (k = 0; k < 4; k = k + 1) begin
            ev[k] = 0; hi[k] = 0; cap[k] = 8'h00; dvd[k] = 1'b0;
        end

        $display("==== TB_SPI_Slave ====");

        // Release reset.
        repeat (5) @(posedge clk);
        rst = 1'b1;
        repeat (5) @(posedge clk);

        // ==================================================================
        // Test 1: single byte, MSB-first, single-cycle DV (mode 0)
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        spi_send(1'b0, 1'b0, 8'hA5);
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(1, (cap[0] === 8'hA5) && (ev[0] == 1) && (hi[0] == 1),
              "0xA5 received MSB-first, one-cycle DV");

        // ==================================================================
        // Test 2: a different byte
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        spi_send(1'b0, 1'b0, 8'h3C);
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(2, (cap[0] === 8'h3C) && (ev[0] == 1) && (hi[0] == 1),
              "0x3C received MSB-first, one-cycle DV");

        // ==================================================================
        // Test 3: multiple bytes within ONE CS-low window
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        spi_send(1'b0, 1'b0, 8'h11);
        spi_send(1'b0, 1'b0, 8'h22);
        spi_send(1'b0, 1'b0, 8'h33);
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(3, (ev[0] == 3) && (rxlog0[0] === 8'h11) &&
                 (rxlog0[1] === 8'h22) && (rxlog0[2] === 8'h33),
              "three bytes in one CS window, in order");

        // ==================================================================
        // Test 4: back-to-back bytes (no inter-byte idle beyond the clocking)
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        spi_send(1'b0, 1'b0, 8'hF0);
        spi_send(1'b0, 1'b0, 8'h0F);
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(4, (ev[0] == 2) && (rxlog0[0] === 8'hF0) && (rxlog0[1] === 8'h0F),
              "back-to-back 0xF0,0x0F both received");

        // ==================================================================
        // Test 5: SCK glitch mid-byte is rejected
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        // 0xB6 = 1011_0110, glitch injected after the 2nd real bit
        m0_bit(1'b1);        // b7
        m0_bit(1'b0);        // b6
        sck_glitch();        // spurious SCK pulse -> must NOT sample a bit
        m0_bit(1'b1);        // b5
        m0_bit(1'b1);        // b4
        m0_bit(1'b0);        // b3
        m0_bit(1'b1);        // b2
        m0_bit(1'b1);        // b1
        m0_bit(1'b0);        // b0
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(5, (cap[0] === 8'hB6) && (ev[0] == 1) && (hi[0] == 1),
              "SCK glitch ignored; 0xB6 intact, one DV");

        // ==================================================================
        // Test 6: CS-deassert glitch mid-frame is rejected
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        // 0x4D = 0100_1101, CS glitch after 4 bits
        m0_bit(1'b0);        // b7
        m0_bit(1'b1);        // b6
        m0_bit(1'b0);        // b5
        m0_bit(1'b0);        // b4
        cs_glitch();         // brief CS deassert -> must NOT break the frame
        m0_bit(1'b1);        // b3
        m0_bit(1'b1);        // b2
        m0_bit(1'b0);        // b1
        m0_bit(1'b1);        // b0
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(6, (cap[0] === 8'h4D) && (ev[0] == 1) && (hi[0] == 1),
              "CS glitch ignored; 0x4D intact, one DV");

        // ==================================================================
        // Test 7: slow/noisy (bouncing) SCK edges
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        // 0x9C = 1001_1100
        m0_bit_noisy(1'b1);  // b7
        m0_bit_noisy(1'b0);  // b6
        m0_bit_noisy(1'b0);  // b5
        m0_bit_noisy(1'b1);  // b4
        m0_bit_noisy(1'b1);  // b3
        m0_bit_noisy(1'b1);  // b2
        m0_bit_noisy(1'b0);  // b1
        m0_bit_noisy(1'b0);  // b0
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(7, (cap[0] === 8'h9C) && (ev[0] == 1) && (hi[0] == 1),
              "noisy/bouncing edges; 0x9C received, one DV");

        // ==================================================================
        // Test 8: CS deassert right after the final SCK edge (last-byte race)
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        spi_send(1'b0, 1'b0, 8'h5C);   // returns with SCK idle after last edge
        // Deassert almost immediately (a few i_Clk) after the last edge; the
        // byte was latched on the sample edge and must survive CS-deassert.
        repeat (2) @(posedge clk);
        cs = 1'b1;
        #(SPI_HALF * 4);
        check(8, (cap[0] === 8'h5C) && (ev[0] == 1) && (hi[0] == 1),
              "last byte delivered despite immediate CS deassert");

        // ==================================================================
        // Test 9: every DV pulse over a burst is exactly one i_Clk cycle
        // ==================================================================
        clear_mon(0); n0 = 0;
        cs = 1'b0; #SPI_HALF;
        spi_send(1'b0, 1'b0, 8'hAA);
        spi_send(1'b0, 1'b0, 8'h55);
        spi_send(1'b0, 1'b0, 8'hC3);
        spi_send(1'b0, 1'b0, 8'h7E);
        #(SPI_HALF * 4);
        cs = 1'b1; #(SPI_HALF * 2);
        check(9, (ev[0] == 4) && (hi[0] == 4),
              "4 bytes -> 4 single-cycle DV pulses (hi == ev)");

        // ==================================================================
        // Test 10: active-low reset clears o_RX_DV and o_RX_Byte
        // ==================================================================
        rst = 1'b0;
        repeat (3) @(posedge clk);
        #1;
        check(10, (dv[0] === 1'b0) && (rx0 === 8'h00),
              "reset clears o_RX_DV and o_RX_Byte");
        rst = 1'b1;
        repeat (5) @(posedge clk);

        // ==================================================================
        // Test 11: TX / MISO behavior (mode 0)
        // ==================================================================
        begin : tx_test
            reg tx_ok;
            integer bi;
            tx_ok = 1'b1;

            // Register the TX byte on i_TX_DV.
            @(posedge clk); tx_byte = 8'hB4; tx_dv = 1'b1;
            @(posedge clk); tx_dv = 1'b0;
            repeat (2) @(posedge clk);

            // CS high -> MISO tri-stated.
            #1;
            if (miso0 !== 1'bz) tx_ok = 1'b0;

            // Assert CS: MSB (bit 7 of 0xB4 = 1) preloaded before the first
            // clock edge.
            sck = 1'b0; mosi = 1'b0;
            cs  = 1'b0; #SPI_HALF;
            if (miso0 !== 1'b1) tx_ok = 1'b0;

            // Clock out MSB-first; slave shifts on the falling (trailing) edge.
            for (bi = 7; bi >= 0; bi = bi - 1) begin
                #1;
                if (miso0 !== ((8'hB4 >> bi) & 1'b1)) tx_ok = 1'b0;
                sck = 1'b1; #SPI_HALF;   // master would sample here
                sck = 1'b0; #SPI_HALF;   // slave advances to next bit
            end

            cs = 1'b1; #SPI_HALF;
            #1;
            if (miso0 !== 1'bz) tx_ok = 1'b0;   // tri-stated again

            check(11, tx_ok,
                  "MISO tri-state/preload/MSB-first shift, TX byte registered");
        end

        // ==================================================================
        // Tests 12-14: non-zero SPI_MODE sampling on the correct edge
        // ==================================================================
        mode_smoke(1, 1'b0, 1'b1, 8'h93);
        check(12, (cap[1] === 8'h93) && (ev[1] == 1) && (hi[1] == 1),
              "SPI_MODE 1 samples on trailing edge (0x93)");

        mode_smoke(2, 1'b1, 1'b0, 8'h5A);
        check(13, (cap[2] === 8'h5A) && (ev[2] == 1) && (hi[2] == 1),
              "SPI_MODE 2 samples on leading edge (0x5A)");

        mode_smoke(3, 1'b1, 1'b1, 8'hC7);
        check(14, (cap[3] === 8'hC7) && (ev[3] == 1) && (hi[3] == 1),
              "SPI_MODE 3 samples on trailing edge (0xC7)");

        // ==================================================================
        // Summary
        // ==================================================================
        $display("");
        $display("================================");
        $display("Results: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("================================");
        $finish;
    end

endmodule
