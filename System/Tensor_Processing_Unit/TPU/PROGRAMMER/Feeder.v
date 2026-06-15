/* 
 * File: Feeder.v
 * Author: Gabe Litteken
 * Date: 2026-06-15
 * 
 * Implements the FETCH step of the CPU instruction cycle. Retrieves
 * instructions from Program_Memory one at a time using a program counter
 * register (pc), checks each instruction for the syscall termination
 * opcode, and forwards non-syscall instructions to the Controller,
 * synchronizing via the controller_idle handshake signal.
 */
module Feeder (

    input              i_clk,
    input              i_trst,

    // Program Memory read interface
    input  [127:0]     i_pm_q,
    output reg [9:0]   o_pm_address,
    output wire        o_pm_wren,

    // Controller interface
    input              i_controller_idle,
    output reg [127:0] o_instruction,
    output reg         o_controller_start

    // ---------- PARAMETERS ---------- //
    // ---------- END PARAMETERS ---------- //

    // ---------- DEBUG ---------- //
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
parameter [9:0] PM_MAX_ADDRESS = 10'd1023;

parameter [3:0]
    STATE_ERROR = 4'd0,
    START       = 4'd1,
    FETCH       = 4'd2,
    WAIT_PM     = 4'd3,
    CHECK       = 4'd4,
    FORWARD     = 4'd5,
    WAIT_CTRL   = 4'd6,
    DONE        = 4'd7,
    FEED_ERROR  = 4'd8;

// syscall opcode per ISA (bits 127:124 = 4'b0000)
parameter [3:0] SYSCALL_OPCODE = 4'b0000;
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// Feeder never writes to program memory
assign o_pm_wren = 1'b0;

// Program counter
reg [9:0] pc;

// State machine registers
reg [3:0] S, NS;

// State machine driver
always @ (posedge i_clk or negedge i_trst)
begin
    if (i_trst == 1'b0)
    begin
        S <= START;
    end
    else
    begin
        S <= NS;
    end
end

// Next state logic
always @ (*)
begin
    case (S)
        START:
            NS = FETCH;
        FETCH:
            NS = WAIT_PM;
        WAIT_PM:
            NS = CHECK;
        CHECK:
            NS = (i_pm_q[127:124] == SYSCALL_OPCODE) ? DONE : FORWARD;
        FORWARD:
            NS = WAIT_CTRL;
        WAIT_CTRL:
            // When controller is idle: increment pc or raise FEED_ERROR if at end
            NS = (i_controller_idle == 1'b1) ?
                 ((pc == PM_MAX_ADDRESS) ? FEED_ERROR : FETCH) :
                 WAIT_CTRL;
        DONE:
            NS = DONE;
        FEED_ERROR:
            NS = FEED_ERROR;
        default:
            NS = STATE_ERROR;
    endcase
end

// Execution logic
always @ (posedge i_clk or negedge i_trst)
begin
    if (i_trst == 1'b0)
    begin
        pc                 <= 10'd0;
        o_pm_address       <= 10'd0;
        o_instruction      <= 128'd0;
        o_controller_start <= 1'b0;
    end
    else
    begin
        case (S)
            FETCH:
            begin
                // Present the current program counter address to program memory;
                // the registered RAM output will be valid in CHECK (two cycles later)
                o_pm_address <= pc;
            end
            CHECK:
            begin
                // Latch instruction so it remains stable through FORWARD/WAIT_CTRL
                o_instruction <= i_pm_q;
            end
            FORWARD:
            begin
                o_controller_start <= 1'b1;
            end
            WAIT_CTRL:
            begin
                o_controller_start <= 1'b0;
                // Increment pc when controller acknowledges, guard against overflow
                if (i_controller_idle == 1'b1 && pc != PM_MAX_ADDRESS)
                begin
                    pc <= pc + 10'd1;
                end
            end
            default:;
        endcase
    end
end

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
