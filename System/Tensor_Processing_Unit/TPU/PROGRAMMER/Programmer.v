/*
 * File: Programmer.v
 * Author: Gabe Litteken
 * Date: 2026-06-30
 *
 * Central controller for all device IO. Decodes the Functional TPU Message
 * Protocol (v0.2) from the SPI input buffer and also exposes device-level
 * input/output. The FSM is organized into six meta-states:
 *
 *   IDLE            - await an SPI header, a device input, or program end.
 *   PROGRAM         - FLASH header: overwrite program memory and write
 *                     weights/inputs to device memory (MEM/PROGRAM commands).
 *   DEVICE_INPUT    - device mode: latch i_input_data into 0x1 buffer addr 0.
 *   DEVICE_OUTPUT   - device mode: emit the first 0x1-buffer value on end.
 *   EXTERNAL_INPUT  - external mode: INPUT header writes a new input to memory
 *                     (unified address routing; PROGRAM command invalid).
 *   EXTERNAL_OUTPUT - external mode: emit the first 10 0x1-buffer values on end,
 *                     pacing each value with the i_device_ready handshake.
 *
 * o_program (active LOW) hands the device memory bus to the Programmer.
 * o_tpu_rst (active LOW) holds the TPU processing internals in reset while
 * memory is being written; it is left HIGH during output so a finished program
 * is not disturbed.
 */
module Programmer (

    input              i_clk,
    input              i_rst,

    // Device-level IO
    input              i_mode_select,       // LOW = device mode, HIGH = external mode
    input              i_input_data_valid,  // one-cycle pulse: i_input_data valid
    input              i_device_ready,      // one-cycle pulse: consumer ready for next output
    input      [7:0]   i_input_data,        // device-mode input value
    output reg [7:0]   o_output_data,       // output value (both modes)
    output reg         o_output_data_valid, // one-cycle pulse: o_output_data valid

    // Feeder interface
    input              i_end_reached,       // one-cycle pulse: end instruction fetched

    // Program Memory write interface
    output reg [127:0] o_pm_data,
    output reg [9:0]   o_pm_address,
    output reg         o_pm_wren,

    // Weight Memory port a write interface
    output reg [7:0]   o_wm_data_a,
    output reg [15:0]  o_wm_address_a,
    output reg         o_wm_wren_a,

    // 0x1 Buffer port a write/read interface
    output reg [7:0]   o_buf_data_a,
    output reg [15:0]  o_buf_address_a,
    output reg         o_buf_wren_a,
    input      [7:0]   i_buf_q_a,           // 0x1 buffer read data (port a)

    // Programming / reset indicators
    output reg         o_program,           // LOW: Programmer owns device memory bus
    output reg         o_tpu_rst,           // LOW: resets TPU processing internals

    // ---------- PARAMETERS ---------- //
    // ---------- END PARAMETERS ---------- //

    // SPI physical interface signals
    input              i_SPI_Clk,
    output wire        o_SPI_MISO,
    input              i_SPI_MOSI,
    input              i_SPI_SS

    // ---------- DEBUG ---------- //
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
// Functional TPU Message Protocol v0.2 function codes (single-byte ASCII).
parameter [7:0]
    FLASH_CODE = 8'h55,  // 'U' header: program the entire peripheral
    INPUT_CODE = 8'h49,  // 'I' header: write a new input
    MEM_CODE   = 8'h4D,  // 'M' command: write bytes to device memory
    PROG_CODE  = 8'h50,  // 'P' command: write a 128-bit instruction
    STOP_CODE  = 8'h53;  // 'S' trailer: end of transmission

// Number of clock cycles to spend in each SPI WAIT state after asserting rdreq
// before reading spi_q. Must be >= 1 to match the registered output latency of
// the SPI input buffer.
parameter integer BUFFER_LATENCY = 1;

// Number of output values emitted in external mode.
parameter integer EXT_OUTPUT_COUNT = 10;

parameter [5:0]
    START_STATE         = 6'd0,
    // IDLE meta-state: scan SPI for a header / await input or end
    IDLE_CHECK          = 6'd1,
    IDLE_POP            = 6'd2,
    IDLE_WAIT           = 6'd3,
    IDLE_EXEC           = 6'd4,
    // Command-code read (shared by PROGRAM and EXTERNAL_INPUT)
    FC_IDLE             = 6'd5,
    FC_POP              = 6'd6,
    FC_WAIT             = 6'd7,
    FC_EXEC             = 6'd8,
    // MEM command field reads
    M_LADD_IDLE         = 6'd9,
    M_LADD_POP          = 6'd10,
    M_LADD_WAIT         = 6'd11,
    M_LADD_EXEC         = 6'd12,
    M_UADD_IDLE         = 6'd13,
    M_UADD_POP          = 6'd14,
    M_UADD_WAIT         = 6'd15,
    M_UADD_EXEC         = 6'd16,
    M_LLEN_IDLE         = 6'd17,
    M_LLEN_POP          = 6'd18,
    M_LLEN_WAIT         = 6'd19,
    M_LLEN_EXEC         = 6'd20,
    M_ULEN_IDLE         = 6'd21,
    M_ULEN_POP          = 6'd22,
    M_ULEN_WAIT         = 6'd23,
    M_ULEN_EXEC         = 6'd24,
    M_DATA_IDLE         = 6'd25,
    M_DATA_POP          = 6'd26,
    M_DATA_WAIT         = 6'd27,
    M_DATA_EXEC         = 6'd28,
    M_WRITE             = 6'd29,
    // PROGRAM command instruction byte reads
    P_BYTE_IDLE         = 6'd30,
    P_BYTE_POP          = 6'd31,
    P_BYTE_WAIT         = 6'd32,
    P_BYTE_EXEC         = 6'd33,
    P_WRITE             = 6'd34,
    // DEVICE_INPUT meta-state
    D_IN_START          = 6'd35,
    D_IN_WRITE          = 6'd36,
    D_IN_DONE           = 6'd37,
    // DEVICE_OUTPUT meta-state
    D_OUT_START         = 6'd38,
    D_OUT_WAIT1         = 6'd39,
    D_OUT_WAIT2         = 6'd40,
    D_OUT_FWD           = 6'd41,
    D_OUT_DONE          = 6'd42,
    // EXTERNAL_OUTPUT meta-state
    X_OUT_START         = 6'd43,
    X_OUT_ADDR          = 6'd44,
    X_OUT_WAIT1         = 6'd45,
    X_OUT_WAIT2         = 6'd46,
    X_OUT_FWD           = 6'd47,
    X_OUT_VALID         = 6'd48,
    X_OUT_WAIT_RDY      = 6'd49,
    X_OUT_INC           = 6'd50,
    X_OUT_DONE          = 6'd51,
    // Error states (unrecoverable; exit only via i_rst LOW)
    STATE_ERROR         = 6'd52,
    COM_ERROR           = 6'd53,
    WRITE_ERROR         = 6'd54,
    PROG_OVERFLOW_ERROR = 6'd55;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

wire       spi_empty;
wire [7:0] spi_q;
reg        spi_rdreq;

SPI_Interface spi_if (
    .clk      (i_clk),
    .rst      (i_rst),
    .i_rdreq  (spi_rdreq),
    .o_empty  (spi_empty),
    .o_full   (),
    .o_q      (spi_q),
    .i_SPI_Clk(i_SPI_Clk),
    .o_SPI_MISO(o_SPI_MISO),
    .i_SPI_MOSI(i_SPI_MOSI),
    .i_SPI_SS (i_SPI_SS)
);

reg [5:0] S, NS;

reg [15:0]  mem_addr;    // current 16-bit ISA write address
reg [15:0]  mem_len;     // remaining bytes to write in MEM sequence
reg         mem_target;  // 0 = Weight_Memory, 1 = 0x1 Buffer
reg [7:0]   data_byte;   // latched data byte for MEM writes
reg [127:0] instr_reg;   // instruction accumulator for PROGRAM writes
reg [3:0]   byte_count;  // byte index within a PROGRAM instruction (0-15)
reg [10:0]  pc;          // program-memory write pointer (11 bits to detect overflow)
reg [31:0]  wait_count;  // cycles spent in current SPI WAIT state
reg         session_type;// 0 = FLASH/PROGRAM session, 1 = INPUT/EXTERNAL_INPUT session

// Latched device-input request: i_input_data_valid is a one-cycle pulse, so it
// is captured here (with its data) and serviced when the FSM returns to IDLE.
reg         input_pending;
reg [7:0]   input_data_lat;

// Latched program-end: i_end_reached is a one-cycle pulse from the Feeder; it is
// captured here and consumed when an output meta-state is entered.
reg         end_latched;

// Output value index for external mode.
reg [3:0]   out_cnt;

// State machine driver
always @(posedge i_clk or negedge i_rst) begin
    if (i_rst == 1'b0)
        S <= START_STATE;
    else
        S <= NS;
end

// Next state logic
always @(*) begin
    case (S)
        START_STATE: NS = IDLE_CHECK;

        // ----- IDLE meta-state ----------------------------------------------
        // Priority: SPI header (PROGRAM / EXTERNAL_INPUT) > DEVICE_INPUT >
        // DEVICE_OUTPUT / EXTERNAL_OUTPUT.
        IDLE_CHECK: NS = (spi_empty == 1'b0)                       ? IDLE_POP :
                         (input_pending && i_mode_select == 1'b0)  ? D_IN_START :
                         (end_latched)                             ?
                             ((i_mode_select == 1'b1) ? X_OUT_START : D_OUT_START) :
                         IDLE_CHECK;
        IDLE_POP:   NS = IDLE_WAIT;
        IDLE_WAIT:  NS = (wait_count >= BUFFER_LATENCY - 1) ? IDLE_EXEC : IDLE_WAIT;
        IDLE_EXEC:  NS = (spi_q == FLASH_CODE) ? FC_IDLE :
                         (spi_q == INPUT_CODE && i_mode_select == 1'b1) ? FC_IDLE :
                         IDLE_CHECK;   // non-header (or device-mode INPUT): discard

        // ----- Command-code read (PROGRAM / EXTERNAL_INPUT) -----------------
        FC_IDLE:    NS = (spi_empty == 1'b0) ? FC_POP : FC_IDLE;
        FC_POP:     NS = FC_WAIT;
        FC_WAIT:    NS = (wait_count >= BUFFER_LATENCY - 1) ? FC_EXEC : FC_WAIT;
        FC_EXEC:    NS = (spi_q == MEM_CODE)  ? M_LADD_IDLE :
                         (spi_q == PROG_CODE) ?
                             // PROGRAM is only valid inside a FLASH session
                             ((session_type == 1'b0) ?
                                 ((pc >= 11'd1024) ? PROG_OVERFLOW_ERROR : P_BYTE_IDLE) :
                                 COM_ERROR) :
                         (spi_q == STOP_CODE) ? IDLE_CHECK :
                         COM_ERROR;

        // ----- MEM command --------------------------------------------------
        M_LADD_IDLE: NS = (spi_empty == 1'b0) ? M_LADD_POP : M_LADD_IDLE;
        M_LADD_POP:  NS = M_LADD_WAIT;
        M_LADD_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? M_LADD_EXEC : M_LADD_WAIT;
        M_LADD_EXEC: NS = M_UADD_IDLE;

        M_UADD_IDLE: NS = (spi_empty == 1'b0) ? M_UADD_POP : M_UADD_IDLE;
        M_UADD_POP:  NS = M_UADD_WAIT;
        M_UADD_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? M_UADD_EXEC : M_UADD_WAIT;
        M_UADD_EXEC: NS = M_LLEN_IDLE;

        M_LLEN_IDLE: NS = (spi_empty == 1'b0) ? M_LLEN_POP : M_LLEN_IDLE;
        M_LLEN_POP:  NS = M_LLEN_WAIT;
        M_LLEN_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? M_LLEN_EXEC : M_LLEN_WAIT;
        M_LLEN_EXEC: NS = M_ULEN_IDLE;

        M_ULEN_IDLE: NS = (spi_empty == 1'b0) ? M_ULEN_POP : M_ULEN_IDLE;
        M_ULEN_POP:  NS = M_ULEN_WAIT;
        M_ULEN_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? M_ULEN_EXEC : M_ULEN_WAIT;
        M_ULEN_EXEC: NS = (mem_addr == 16'd0)             ? WRITE_ERROR :
                         ({spi_q, mem_len[7:0]} == 16'd0) ? FC_IDLE     :
                         M_DATA_IDLE;

        M_DATA_IDLE: NS = (mem_len == 16'd0) ? FC_IDLE    :
                         (spi_empty == 1'b0) ? M_DATA_POP : M_DATA_IDLE;
        M_DATA_POP:  NS = M_DATA_WAIT;
        M_DATA_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? M_DATA_EXEC : M_DATA_WAIT;
        M_DATA_EXEC: NS = M_WRITE;
        M_WRITE:     NS = M_DATA_IDLE;

        // ----- PROGRAM command ----------------------------------------------
        P_BYTE_IDLE: NS = (spi_empty == 1'b0) ? P_BYTE_POP : P_BYTE_IDLE;
        P_BYTE_POP:  NS = P_BYTE_WAIT;
        P_BYTE_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? P_BYTE_EXEC : P_BYTE_WAIT;
        P_BYTE_EXEC: NS = (byte_count == 4'd15) ? P_WRITE : P_BYTE_IDLE;
        P_WRITE:     NS = FC_IDLE;

        // ----- DEVICE_INPUT meta-state --------------------------------------
        D_IN_START:  NS = D_IN_WRITE;
        D_IN_WRITE:  NS = D_IN_DONE;
        D_IN_DONE:   NS = IDLE_CHECK;

        // ----- DEVICE_OUTPUT meta-state -------------------------------------
        D_OUT_START: NS = D_OUT_WAIT1;
        D_OUT_WAIT1: NS = D_OUT_WAIT2;
        D_OUT_WAIT2: NS = D_OUT_FWD;
        D_OUT_FWD:   NS = D_OUT_DONE;
        D_OUT_DONE:  NS = IDLE_CHECK;

        // ----- EXTERNAL_OUTPUT meta-state -----------------------------------
        X_OUT_START:    NS = X_OUT_ADDR;
        X_OUT_ADDR:     NS = X_OUT_WAIT1;
        X_OUT_WAIT1:    NS = X_OUT_WAIT2;
        X_OUT_WAIT2:    NS = X_OUT_FWD;
        X_OUT_FWD:      NS = X_OUT_VALID;
        X_OUT_VALID:    NS = X_OUT_WAIT_RDY;
        X_OUT_WAIT_RDY: NS = (i_device_ready == 1'b1) ? X_OUT_INC : X_OUT_WAIT_RDY;
        X_OUT_INC:      NS = (out_cnt == EXT_OUTPUT_COUNT - 1) ? X_OUT_DONE : X_OUT_ADDR;
        X_OUT_DONE:     NS = IDLE_CHECK;

        STATE_ERROR:         NS = STATE_ERROR;
        COM_ERROR:           NS = COM_ERROR;
        WRITE_ERROR:         NS = WRITE_ERROR;
        PROG_OVERFLOW_ERROR: NS = PROG_OVERFLOW_ERROR;
        default:             NS = STATE_ERROR;
    endcase
end

// Execution logic
always @(posedge i_clk or negedge i_rst) begin
    if (i_rst == 1'b0) begin
        spi_rdreq           <= 1'b0;
        o_pm_data           <= 128'd0;
        o_pm_address        <= 10'd0;
        o_pm_wren           <= 1'b0;
        o_wm_data_a         <= 8'd0;
        o_wm_address_a      <= 16'd0;
        o_wm_wren_a         <= 1'b0;
        o_buf_data_a        <= 8'd0;
        o_buf_address_a     <= 16'd0;
        o_buf_wren_a        <= 1'b0;
        o_program           <= 1'b1;
        o_tpu_rst           <= 1'b1;
        o_output_data       <= 8'd0;
        o_output_data_valid <= 1'b0;
        mem_addr            <= 16'd0;
        mem_len             <= 16'd0;
        mem_target          <= 1'b0;
        data_byte           <= 8'd0;
        instr_reg           <= 128'd0;
        byte_count          <= 4'd0;
        pc                  <= 11'd0;
        wait_count          <= 32'd0;
        session_type        <= 1'b0;
        input_pending       <= 1'b0;
        input_data_lat      <= 8'd0;
        end_latched         <= 1'b0;
        out_cnt             <= 4'd0;
    end else begin
        // Default: deassert single-cycle strobes; reset wait counter.
        spi_rdreq           <= 1'b0;
        o_pm_wren           <= 1'b0;
        o_wm_wren_a         <= 1'b0;
        o_buf_wren_a        <= 1'b0;
        o_output_data_valid <= 1'b0;
        wait_count          <= 32'd0;

        // Latch one-cycle pulses (placed before the case so a state can consume
        // a latch on the same cycle by overriding the assignment below).
        if (i_end_reached)
            end_latched <= 1'b1;
        if (i_input_data_valid && i_mode_select == 1'b0) begin
            input_pending  <= 1'b1;
            input_data_lat <= i_input_data;
        end

        case (S)
            // ----- IDLE meta-state ------------------------------------------
            IDLE_POP:  spi_rdreq  <= 1'b1;
            IDLE_WAIT: wait_count <= wait_count + 32'd1;
            IDLE_EXEC: begin
                if (spi_q == FLASH_CODE) begin
                    // FLASH: begin a PROGRAM session; overwrite program memory.
                    session_type <= 1'b0;
                    o_program    <= 1'b0;
                    o_tpu_rst    <= 1'b0;
                    pc           <= 11'd0;
                    end_latched  <= 1'b0;
                end else if (spi_q == INPUT_CODE && i_mode_select == 1'b1) begin
                    // INPUT (external mode): begin an EXTERNAL_INPUT session;
                    // the current program is maintained (pc untouched).
                    session_type <= 1'b1;
                    o_program    <= 1'b0;
                    o_tpu_rst    <= 1'b0;
                    end_latched  <= 1'b0;
                end
                // Any other byte (including device-mode INPUT) is discarded.
            end

            // ----- Command-code read ----------------------------------------
            FC_POP:  spi_rdreq  <= 1'b1;
            FC_WAIT: wait_count <= wait_count + 32'd1;
            FC_EXEC: if (spi_q == STOP_CODE) begin
                // Trailer: release the device memory bus and the TPU internals.
                o_program <= 1'b1;
                o_tpu_rst <= 1'b1;
            end

            // ----- MEM command ----------------------------------------------
            M_LADD_POP:  spi_rdreq  <= 1'b1;
            M_LADD_WAIT: wait_count <= wait_count + 32'd1;
            M_LADD_EXEC: mem_addr[7:0]  <= spi_q;

            M_UADD_POP:  spi_rdreq  <= 1'b1;
            M_UADD_WAIT: wait_count <= wait_count + 32'd1;
            M_UADD_EXEC: mem_addr[15:8] <= spi_q;

            M_LLEN_POP:  spi_rdreq  <= 1'b1;
            M_LLEN_WAIT: wait_count <= wait_count + 32'd1;
            M_LLEN_EXEC: mem_len[7:0]   <= spi_q;

            M_ULEN_POP:  spi_rdreq  <= 1'b1;
            M_ULEN_WAIT: wait_count <= wait_count + 32'd1;
            M_ULEN_EXEC: begin
                mem_len[15:8] <= spi_q;
                mem_target    <= (mem_addr == 16'd1) ? 1'b1 : 1'b0;
            end

            M_DATA_POP:  spi_rdreq  <= 1'b1;
            M_DATA_WAIT: wait_count <= wait_count + 32'd1;
            M_DATA_EXEC: data_byte  <= spi_q;

            M_WRITE: begin
                case (mem_target)
                    1'b0: begin
                        // Weight_Memory: ISA addresses 0x0 and 0x1 are reserved,
                        // so ISA address A maps to physical address A-2, matching
                        // the Vector_Processor's read/write mapping (unified
                        // address space).
                        o_wm_wren_a    <= 1'b1;
                        o_wm_address_a <= mem_addr - 16'd2;
                        o_wm_data_a    <= data_byte;
                    end
                    default: begin
                        // 0x1 Buffer: ISA address 0x0001 -> internal address 0x0000
                        o_buf_wren_a    <= 1'b1;
                        o_buf_address_a <= mem_addr - 16'd1;
                        o_buf_data_a    <= data_byte;
                    end
                endcase
                mem_addr <= mem_addr + 16'd1;
                mem_len  <= mem_len  - 16'd1;
            end

            // ----- PROGRAM command ------------------------------------------
            P_BYTE_POP:  spi_rdreq  <= 1'b1;
            P_BYTE_WAIT: wait_count <= wait_count + 32'd1;
            P_BYTE_EXEC: begin
                // Shift new byte into MSB; first received byte ends at [7:0]
                // (LSB first per protocol).
                instr_reg  <= {spi_q, instr_reg[127:8]};
                byte_count <= byte_count + 4'd1;
            end
            P_WRITE: begin
                o_pm_wren    <= 1'b1;
                o_pm_address <= pc[9:0];
                o_pm_data    <= instr_reg;
                pc           <= pc + 11'd1;
                byte_count   <= 4'd0;
            end

            // ----- DEVICE_INPUT meta-state ----------------------------------
            D_IN_START: begin
                o_program     <= 1'b0;
                o_tpu_rst     <= 1'b0;
                input_pending <= 1'b0;
                end_latched   <= 1'b0;   // a fresh run will re-assert end
            end
            D_IN_WRITE: begin
                // Write the latched input to the first 0x1-buffer address.
                o_buf_wren_a    <= 1'b1;
                o_buf_address_a <= 16'd0;
                o_buf_data_a    <= input_data_lat;
            end
            D_IN_DONE: begin
                o_program <= 1'b1;
                o_tpu_rst <= 1'b1;
            end

            // ----- DEVICE_OUTPUT meta-state ---------------------------------
            D_OUT_START: begin
                o_program       <= 1'b0;   // take the bus to read the 0x1 buffer
                o_buf_address_a <= 16'd0;
                end_latched     <= 1'b0;
            end
            D_OUT_FWD: begin
                o_output_data       <= i_buf_q_a;
                o_output_data_valid <= 1'b1;
            end
            D_OUT_DONE: o_program <= 1'b1;

            // ----- EXTERNAL_OUTPUT meta-state -------------------------------
            X_OUT_START: begin
                o_program   <= 1'b0;
                out_cnt     <= 4'd0;
                end_latched <= 1'b0;
            end
            X_OUT_ADDR: o_buf_address_a <= {12'd0, out_cnt};
            X_OUT_FWD: begin
                o_output_data       <= i_buf_q_a;
                o_output_data_valid <= 1'b1;
            end
            X_OUT_INC:  out_cnt   <= out_cnt + 4'd1;
            X_OUT_DONE: o_program <= 1'b1;

            default: ;
        endcase
    end
end

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
