/*
 * File: Pooler.v
 * Author: Gabe Litteken
 * Date: 2026-07-16
 *
 * Streaming reducer that implements the POOL instruction functions from the
 * Functional TPU ISA. Unlike the ALU and Activator (one output per input), the
 * Pooler folds a stream of input values into a single accumulator and emits one
 * pooled value per pooling window.
 *
 * While i_enable is HIGH the module folds one input per clock into the
 * accumulator using the function-selected reduction operator (for max, a signed
 * comparator whose identity is the minimum representable value, -128). On the
 * final value of a window the Vector_Processor asserts i_window_end coincident
 * with i_enable: the module folds that final value, writes the resulting pooled
 * value to the vector buffer (o_write pulsed HIGH for one cycle with o_data
 * holding the result), and resets the accumulator to the reduction identity for
 * the next window. The pooled result is one of the input values already in the
 * memory datatype, so no requantization is performed and overflow cannot occur.
 * Buffer writing is suppressed when the function is NO OP.
 */
module Pooler (

    input i_clk,
    input i_trst,

    // control signals
    input i_enable,
    input i_clear,
    input i_window_end,
    output wire o_write,
    input [2:0] i_pooler_op,

    // data signals
    input [7:0] i_data,
    output reg [7:0] o_data

    // ---------- PARAMETERS ---------- //
    // ---------- END PARAMETERS ---------- //

    // ---------- DEBUG ---------- //
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
parameter [2:0]
    MAX  = 3'h0,   // max pooling (POOL funct3 0x0 per ISA)
    NOOP = 3'h7;

// Reduction identity for max is the minimum representable signed 8-bit value.
// Padding elements are substituted with this value upstream so they can never
// win the comparison.
localparam signed [7:0] MIN_VALUE = 8'h80;   // -128
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// Running reduction accumulator. Holds the reduction of all window values seen
// so far (excluding the value present on i_data during the current cycle, which
// is folded in on the next posedge or emitted combinationally on window_end).
reg signed [7:0] acc;

// Fold the current input into the running accumulator for the selected
// function. For max this is a signed comparator; other functions hold.
function signed [7:0] fold;
    input signed [7:0] accumulator;
    input signed [7:0] value;
    input [2:0]        op;
    begin
        case (op)
            MAX:     fold = (accumulator > value) ? accumulator : value;
            default: fold = accumulator;   // NOOP / control error: no reduction
        endcase
    end
endfunction

// A window is emitted exactly once, on the cycle its final value is enabled.
// Suppressed for NO OP so a non-pooling operation cannot push data to the
// buffer.
assign o_write = i_enable & i_window_end & (i_pooler_op != NOOP);

// o_data is combinational so that, on the window_end cycle, it holds the fully
// reduced window (running accumulator folded with the final input value) at the
// same time o_write is HIGH, matching the vector buffer's single-cycle capture.
always @ (*)
begin
    case (i_pooler_op)
        MAX:     o_data = fold(acc, $signed(i_data), MAX);
        NOOP:    o_data = 8'd98;   // indicates NOOP
        default: o_data = 8'd29;   // indicates control error
    endcase
end

// Accumulator update. Reset to the reduction identity on trst or clear (the
// inter-operation flush). While enabled, fold each value; on the final value of
// a window (window_end) reset to the identity for the next window, since that
// value has already been emitted combinationally on o_data this cycle.
always @ (posedge i_clk or negedge i_trst)
begin
    if (i_trst == 1'b0)
        acc <= MIN_VALUE;
    else if (i_clear == 1'b1)
        acc <= MIN_VALUE;
    else if (i_enable == 1'b1)
    begin
        if (i_window_end == 1'b1)
            acc <= MIN_VALUE;
        else
            acc <= fold(acc, $signed(i_data), i_pooler_op);
    end
end
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
