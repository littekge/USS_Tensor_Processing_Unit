/* 
 * File: Programmer.v
 * Author: Gabe Litteken
 * Date: 2026-06-12
 * 
 * Reads bytes from the SPI input buffer, decodes the Functional TPU Messaging
 * Protocol, and writes data to Program_Memory, Weight_Memory, or the 0x1
 * Buffer. Asserts o_program LOW during an active programming session.
 */
module Programmer (

    input              i_clk,
    input              i_rst,

    // Program Memory write interface
    output reg [127:0] o_pm_data,
    output reg [9:0]   o_pm_address,
    output reg         o_pm_wren,

    // Weight Memory port a write interface
    output reg [7:0]   o_wm_data_a,
    output reg [15:0]  o_wm_address_a,
    output reg         o_wm_wren_a,

    // 0x1 Buffer port a write interface
    output reg [7:0]   o_buf_data_a,
    output reg [15:0]  o_buf_address_a,
    output reg         o_buf_wren_a,

    // Programming active indicator (active LOW)
    output reg         o_program,

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
parameter [7:0]
    START_CODE = 8'h55,
    STOP_CODE  = 8'h53,
    MEM_CODE   = 8'h4D,
    PROG_CODE  = 8'h50;

// Number of clock cycles to spend in each WAIT state after asserting rdreq
// before reading spi_q. Must be >= 1 to match the registered output latency
// of the SPI input buffer.
parameter integer BUFFER_LATENCY = 1;

parameter [5:0]
    START_STATE         = 6'd0,
    SCAN_IDLE           = 6'd1,
    SCAN_POP            = 6'd2,
    SCAN_WAIT           = 6'd3,
    SCAN_EXEC           = 6'd4,
    FC_IDLE             = 6'd5,
    FC_POP              = 6'd6,
    FC_WAIT             = 6'd7,
    FC_EXEC             = 6'd8,
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
    P_BYTE_IDLE         = 6'd30,
    P_BYTE_POP          = 6'd31,
    P_BYTE_WAIT         = 6'd32,
    P_BYTE_EXEC         = 6'd33,
    P_WRITE             = 6'd34,
    STATE_ERROR         = 6'd35,
    COM_ERROR           = 6'd36,
    WRITE_ERROR         = 6'd37,
    PROG_OVERFLOW_ERROR = 6'd38;
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
reg [10:0]  pc;          // program counter (11 bits to detect overflow past 1023)
reg [31:0]  wait_count;  // cycles spent in current WAIT state

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
        START_STATE: NS = SCAN_IDLE;

        SCAN_IDLE:   NS = (spi_empty == 1'b0) ? SCAN_POP  : SCAN_IDLE;
        SCAN_POP:    NS = SCAN_WAIT;
        SCAN_WAIT:   NS = (wait_count >= BUFFER_LATENCY - 1) ? SCAN_EXEC  : SCAN_WAIT;
        SCAN_EXEC:   NS = (spi_q == START_CODE) ? FC_IDLE : SCAN_IDLE;

        FC_IDLE:     NS = (spi_empty == 1'b0) ? FC_POP : FC_IDLE;
        FC_POP:      NS = FC_WAIT;
        FC_WAIT:     NS = (wait_count >= BUFFER_LATENCY - 1) ? FC_EXEC : FC_WAIT;
        FC_EXEC:     NS = (spi_q == MEM_CODE)  ? M_LADD_IDLE :
                         (spi_q == PROG_CODE)  ? ((pc >= 11'd1024) ? PROG_OVERFLOW_ERROR
                                                                     : P_BYTE_IDLE) :
                         (spi_q == STOP_CODE)  ? SCAN_IDLE :
                         COM_ERROR;

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
        M_ULEN_EXEC: NS = (mem_addr == 16'd0)               ? WRITE_ERROR :
                         ({spi_q, mem_len[7:0]} == 16'd0)   ? FC_IDLE     :
                         M_DATA_IDLE;

        M_DATA_IDLE: NS = (mem_len == 16'd0)   ? FC_IDLE    :
                         (spi_empty == 1'b0)   ? M_DATA_POP : M_DATA_IDLE;
        M_DATA_POP:  NS = M_DATA_WAIT;
        M_DATA_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? M_DATA_EXEC : M_DATA_WAIT;
        M_DATA_EXEC: NS = M_WRITE;
        M_WRITE:     NS = M_DATA_IDLE;

        P_BYTE_IDLE: NS = (spi_empty == 1'b0) ? P_BYTE_POP : P_BYTE_IDLE;
        P_BYTE_POP:  NS = P_BYTE_WAIT;
        P_BYTE_WAIT: NS = (wait_count >= BUFFER_LATENCY - 1) ? P_BYTE_EXEC : P_BYTE_WAIT;
        P_BYTE_EXEC: NS = (byte_count == 4'd15) ? P_WRITE : P_BYTE_IDLE;
        P_WRITE:     NS = FC_IDLE;

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
        spi_rdreq       <= 1'b0;
        o_pm_data       <= 128'd0;
        o_pm_address    <= 10'd0;
        o_pm_wren       <= 1'b0;
        o_wm_data_a     <= 8'd0;
        o_wm_address_a  <= 16'd0;
        o_wm_wren_a     <= 1'b0;
        o_buf_data_a    <= 8'd0;
        o_buf_address_a <= 16'd0;
        o_buf_wren_a    <= 1'b0;
        o_program       <= 1'b1;
        mem_addr        <= 16'd0;
        mem_len         <= 16'd0;
        mem_target      <= 1'b0;
        data_byte       <= 8'd0;
        instr_reg       <= 128'd0;
        byte_count      <= 4'd0;
        pc              <= 11'd0;
        wait_count      <= 32'd0;
    end else begin
        // Default: deassert single-cycle strobes; reset wait counter
        spi_rdreq    <= 1'b0;
        o_pm_wren    <= 1'b0;
        o_wm_wren_a  <= 1'b0;
        o_buf_wren_a <= 1'b0;
        wait_count   <= 32'd0;

        case (S)
            SCAN_POP:    spi_rdreq  <= 1'b1;
            SCAN_WAIT:   wait_count <= wait_count + 32'd1;
            SCAN_EXEC:   if (spi_q == START_CODE) o_program <= 1'b0;

            FC_POP:      spi_rdreq  <= 1'b1;
            FC_WAIT:     wait_count <= wait_count + 32'd1;
            FC_EXEC:     if (spi_q == STOP_CODE)  o_program <= 1'b1;

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
                        // Weight_Memory: ISA address maps directly to memory address
                        o_wm_wren_a    <= 1'b1;
                        o_wm_address_a <= mem_addr;
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

            P_BYTE_POP:  spi_rdreq  <= 1'b1;
            P_BYTE_WAIT: wait_count <= wait_count + 32'd1;
            P_BYTE_EXEC: begin
                // Shift new byte into MSB; first received byte ends at [7:0] (LSB first per protocol)
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

            default: ;
        endcase
    end
end

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
