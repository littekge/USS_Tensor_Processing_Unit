///////////////////////////////////////////////////////////////////////////////
// File:   SPI_Slave.v
// Author: Functional TPU project
// Date:   2026-07-14
//
// Description: SPI (Serial Peripheral Interface) Slave — fully SYNCHRONOUS
//              OVERSAMPLING receiver in the i_Clk (50 MHz) domain.
//
//              This is NO LONGER the nandland design. The previous version
//              used the raw SPI clock (i_SPI_Clk) as a real clock and the raw
//              chip-select (i_SPI_CS_n) as an asynchronous reset, and crossed
//              a "byte done" flag between the SPI and FPGA clock domains. On
//              hardware that exposed three failure modes when SPI edges were
//              slow/noisy (passive 5V->3.3V level shifting): glitch-triggered
//              spurious clocking, a last-byte race where CS deassert
//              async-cleared the byte before it crossed domains, and general
//              CDC metastability. This rewrite removes all three: i_SPI_Clk,
//              i_SPI_MOSI, and i_SPI_CS_n are treated purely as DATA inputs,
//              never as clocks or resets. Everything is clocked by i_Clk.
//
//              How it works:
//                - 2-stage FF synchronizers bring the three SPI inputs into
//                  the i_Clk domain.
//                - SCK and CS are DEBOUNCED (a level change is accepted only
//                  after DEBOUNCE_N consecutive equal samples), rejecting
//                  glitches and slow-edge bounce.
//                - The debounced SCK is edge-detected; SPI_MODE (CPOL/CPHA)
//                  selects the sample edge (capture MOSI) and the shift edge
//                  (advance MISO). On each sample edge the synchronized MOSI
//                  bit is shifted MSB-first into an 8-bit register; the 8th
//                  bit latches o_RX_Byte and pulses o_RX_DV for one i_Clk.
//                - CS framing is synchronous: CS-assert resets the RX bit
//                  counter (frame start); CS-deassert NEVER clears received
//                  byte / DV state (this is what fixes the last-byte race). A
//                  partial byte is simply discarded when the next frame starts.
//                - TX/MISO is driven synchronously off the detected shift edge
//                  and tri-stated when CS is (synchronized) high.
//
// Interface/behavior UNCHANGED from the previous version (drop-in):
//                - Ports and the SPI_MODE parameter are identical.
//                - o_RX_DV is a single i_Clk-cycle pulse per received byte with
//                  o_RX_Byte (MSB-first) valid on that cycle.
//                - Multiple / back-to-back bytes per CS-low window supported.
//                - TX: preload MSB on CS-assert, MSB-first shift on shift edge,
//                  tri-state on CS-high, register i_TX_Byte on i_TX_DV.
//                - i_Rst_L active-low reset; full SPI_MODE 0-3 semantics.
//
// Timing note: i_Clk must oversample each SCK half-period by at least the
//              synchronizer + debounce depth (2 + DEBOUNCE_N ~= 5 i_Clk). At
//              50 MHz i_Clk with DEBOUNCE_N = 3 this bounds SCK to a few MHz
//              — far above the deployed link's 125 kHz (the AVR SPI floor).
//
// Parameters:  SPI_MODE, 0-3 (unchanged):
//              Mode | Clock Polarity (CPOL) | Clock Phase (CPHA)
//               0   |          0            |        0
//               1   |          0            |        1
//               2   |          1            |        0
//               3   |          1            |        1
//              https://en.wikipedia.org/wiki/Serial_Peripheral_Interface_Bus#Mode_numbers
///////////////////////////////////////////////////////////////////////////////

module SPI_Slave
  #(parameter SPI_MODE = 0)
  (
   // Control/Data Signals,
   input            i_Rst_L,    // FPGA Reset, active low
   input            i_Clk,      // FPGA Clock
   output reg       o_RX_DV,    // Data Valid pulse (1 clock cycle)
   output reg [7:0] o_RX_Byte,  // Byte received on MOSI
   input            i_TX_DV,    // Data Valid pulse to register i_TX_Byte
   input  [7:0]     i_TX_Byte,  // Byte to serialize to MISO.

   // SPI Interface
   input      i_SPI_Clk,
   output wire o_SPI_MISO,
   input      i_SPI_MOSI,
   input      i_SPI_CS_n        // active low
   );

// ---------- PARAMETERS ---------- //
  // A debounced level change is accepted only after this many consecutive
  // equal i_Clk samples. 3 samples (~60 ns at 50 MHz) filter sub-60 ns glitches
  // and the bounce during the ~100 ns resistive-divider edges seen on the
  // deployed 5V->3.3V path, yet are trivially short next to the 125 kHz SCK
  // half-period (~4 us). Raise DEBOUNCE_N if slower/noisier edges appear.
  localparam DEBOUNCE_N = 3;

  // CPOL: clock idle level (0 => idles low, leading edge rising;
  //       1 => idles high, leading edge falling).
  // CPHA: 0 => sample on leading edge; 1 => sample on trailing edge.
  localparam CPOL = (SPI_MODE == 2) || (SPI_MODE == 3);
  localparam CPHA = (SPI_MODE == 1) || (SPI_MODE == 3);
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

  // ----- 2-stage synchronizers (SPI pins -> i_Clk domain) ----- //
  // Reset SCK to its idle level and CS to deasserted so no spurious first
  // edge or frame is manufactured out of reset.
  reg [1:0] r_SCK_Sync;
  reg [1:0] r_MOSI_Sync;
  reg [1:0] r_CS_Sync;

  always @(posedge i_Clk or negedge i_Rst_L)
  begin
    if (~i_Rst_L)
    begin
      r_SCK_Sync  <= CPOL ? 2'b11 : 2'b00;   // idle SCK level
      r_MOSI_Sync <= 2'b00;
      r_CS_Sync   <= 2'b11;      // CS_n high = deasserted
    end
    else
    begin
      r_SCK_Sync  <= {r_SCK_Sync[0],  i_SPI_Clk};
      r_MOSI_Sync <= {r_MOSI_Sync[0], i_SPI_MOSI};
      r_CS_Sync   <= {r_CS_Sync[0],   i_SPI_CS_n};
    end
  end

  // ----- Debounce SCK and CS ----- //
  // r_*_Stable holds the accepted (debounced) level; r_*_Stable_d is its
  // previous value for edge detection. r_*_Cnt counts how long the
  // synchronized sample has continuously differed from the accepted level.
  reg       r_SCK_Stable, r_SCK_Stable_d;
  reg [3:0] r_SCK_Cnt;
  reg       r_CS_Stable,  r_CS_Stable_d;
  reg [3:0] r_CS_Cnt;

  always @(posedge i_Clk or negedge i_Rst_L)
  begin
    if (~i_Rst_L)
    begin
      r_SCK_Stable   <= CPOL ? 1'b1 : 1'b0;
      r_SCK_Stable_d <= CPOL ? 1'b1 : 1'b0;
      r_SCK_Cnt      <= 4'd0;
      r_CS_Stable    <= 1'b1;
      r_CS_Stable_d  <= 1'b1;
      r_CS_Cnt       <= 4'd0;
    end
    else
    begin
      r_SCK_Stable_d <= r_SCK_Stable;
      r_CS_Stable_d  <= r_CS_Stable;

      // SCK debounce
      if (r_SCK_Sync[1] == r_SCK_Stable)
      begin
        r_SCK_Cnt <= 4'd0;                 // no pending change
      end
      else if (r_SCK_Cnt == (DEBOUNCE_N - 1))
      begin
        r_SCK_Stable <= r_SCK_Sync[1];     // change held long enough: accept
        r_SCK_Cnt    <= 4'd0;
      end
      else
      begin
        r_SCK_Cnt <= r_SCK_Cnt + 4'd1;
      end

      // CS debounce
      if (r_CS_Sync[1] == r_CS_Stable)
      begin
        r_CS_Cnt <= 4'd0;
      end
      else if (r_CS_Cnt == (DEBOUNCE_N - 1))
      begin
        r_CS_Stable <= r_CS_Sync[1];
        r_CS_Cnt    <= 4'd0;
      end
      else
      begin
        r_CS_Cnt <= r_CS_Cnt + 4'd1;
      end
    end
  end

  // ----- Debounced edges / levels ----- //
  wire w_SCK_Rise = r_SCK_Stable & ~r_SCK_Stable_d;
  wire w_SCK_Fall = ~r_SCK_Stable & r_SCK_Stable_d;

  // Leading edge polarity is set by CPOL; sample vs shift edge by CPHA.
  wire w_Leading  = CPOL ? w_SCK_Fall : w_SCK_Rise;
  wire w_Trailing = CPOL ? w_SCK_Rise : w_SCK_Fall;
  wire w_Sample_Edge = CPHA ? w_Trailing : w_Leading;
  wire w_Shift_Edge  = CPHA ? w_Leading  : w_Trailing;

  wire w_CS_Active   = ~r_CS_Stable;                   // CS asserted (low)
  wire w_CS_Assert   = ~r_CS_Stable & r_CS_Stable_d;   // CS_n falling: frame start
  // (CS_n rising / deassert is intentionally NOT used to clear RX state.)

  // ----- RX datapath (MSB-first) ----- //
  reg [2:0] r_RX_Bit_Count;
  reg [7:0] r_RX_Shift;

  always @(posedge i_Clk or negedge i_Rst_L)
  begin
    if (~i_Rst_L)
    begin
      r_RX_Bit_Count <= 3'd0;
      r_RX_Shift     <= 8'd0;
      o_RX_Byte      <= 8'd0;
      o_RX_DV        <= 1'b0;
    end
    else
    begin
      o_RX_DV <= 1'b0;  // default: DV is a single-cycle pulse

      if (w_CS_Assert)
      begin
        // New frame: restart bit counter, discarding any partial byte.
        r_RX_Bit_Count <= 3'd0;
      end
      else if (w_CS_Active & w_Sample_Edge)
      begin
        // First sampled bit ends up in bit 7 after eight left shifts.
        r_RX_Shift <= {r_RX_Shift[6:0], r_MOSI_Sync[1]};

        if (r_RX_Bit_Count == 3'd7)
        begin
          o_RX_Byte      <= {r_RX_Shift[6:0], r_MOSI_Sync[1]};
          o_RX_DV        <= 1'b1;
          r_RX_Bit_Count <= 3'd0;
        end
        else
        begin
          r_RX_Bit_Count <= r_RX_Bit_Count + 3'd1;
        end
      end
    end
  end

  // ----- TX byte register (loaded on i_TX_DV) ----- //
  reg [7:0] r_TX_Byte;

  always @(posedge i_Clk or negedge i_Rst_L)
  begin
    if (~i_Rst_L)
      r_TX_Byte <= 8'h00;
    else if (i_TX_DV)
      r_TX_Byte <= i_TX_Byte;
  end

  // ----- TX/MISO datapath (MSB-first, synchronous) ----- //
  // Preload the MSB the moment CS asserts, then advance on each shift edge.
  reg [2:0] r_TX_Bit_Index;
  reg       r_MISO_Bit;

  always @(posedge i_Clk or negedge i_Rst_L)
  begin
    if (~i_Rst_L)
    begin
      r_TX_Bit_Index <= 3'd7;
      r_MISO_Bit     <= 1'b0;
    end
    else if (w_CS_Assert)
    begin
      r_TX_Bit_Index <= 3'd7;
      r_MISO_Bit     <= r_TX_Byte[7];   // preload MSB before first clock edge
    end
    else if (w_CS_Active & w_Shift_Edge)
    begin
      r_TX_Bit_Index <= r_TX_Bit_Index - 3'd1;
      r_MISO_Bit     <= r_TX_Byte[r_TX_Bit_Index - 3'd1];
    end
  end

  // Tri-state MISO while CS is (synchronized) high so multiple slaves may
  // share the bus; drive the current MISO bit while selected.
  assign o_SPI_MISO = r_CS_Stable ? 1'bZ : r_MISO_Bit;

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule // SPI_Slave
