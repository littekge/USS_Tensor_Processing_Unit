/* 
 * File: Controller.v
 * Author: Gabe Litteken
 * Date: 2026-06-15
 * 
 * The Controller module is the "CPU" of the Functional TPU. Combined with
 * the Feeder module, it implements the traditional CPU instruction cycle
 * (Fetch-Decode-Execute-Writeback). The Controller decodes instructions
 * from the Feeder and generates control signals for the Vector_Processors,
 * ALU, Activator, and Systolic_Array, managing timing and dataflow between
 * all processing modules.
 */
module Controller (

    input              i_clk,
    input              i_trst,

    // Feeder interface
    input              i_controller_start,
    input  [127:0]     i_instruction,

    // Vector_Processor handshake inputs
    input              i_vector_idle_a,
    input              i_vector_idle_b,

    // Systolic_Array handshake input
    input              i_systolic_array_idle,

    // ---------- PARAMETERS ---------- //
    // ---------- END PARAMETERS ---------- //

    // Feeder handshake and global clear
    output wire        o_controller_idle,
    output wire        o_clear,

    // Vector_Processor_A control outputs
    output wire        o_vector_start_a,
    output reg  [2:0]  o_vect_source_a,
    output reg  [2:0]  o_vect_dest_a,
    output reg  [23:0] o_mem_source_address_a,
    output reg  [23:0] o_mem_dest_address_a,
    output reg  [15:0] o_length_a,
    output reg  [3:0]  o_dim0_a,
    output reg  [3:0]  o_dim1_a,

    // Per-layer dyadic requantization parameters for VP_A (MUL writeback).
    // Only VP_A reads systolic array outputs and requantizes; VP_B never does.
    output reg  [7:0]  o_scale_a,   // M0: unsigned dyadic multiplier
    output reg  [7:0]  o_shift_a,   // n:  unsigned right-shift amount

    // Vector_Processor_B control outputs
    output wire        o_vector_start_b,
    output reg  [2:0]  o_vect_source_b,
    output reg  [2:0]  o_vect_dest_b,
    output reg  [23:0] o_mem_source_address_b,
    output reg  [15:0] o_length_b,
    output reg  [3:0]  o_dim0_b,
    output reg  [3:0]  o_dim1_b,

    // Activator and ALU function selects
    output wire [2:0]  o_activator_funct,
    output wire [2:0]  o_alu_funct,

    // Systolic_Array control output
    output wire        o_systolic_array_start

    // ---------- DEBUG ---------- //
    // ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //

// State machine states
parameter [3:0]
    STATE_ERROR   = 4'd0,
    START         = 4'd1,
    IDLE          = 4'd2,
    CLEAR         = 4'd3,
    EXEC_VP_START = 4'd4,
    EXEC_VP_WAIT  = 4'd5,
    SA_START      = 4'd6,
    SA_WAIT       = 4'd7,
    WB_VP_START   = 4'd8,
    WB_VP_WAIT    = 4'd9,
    DECODE_ERROR  = 4'd10;

// ISA instruction opcodes
parameter [3:0]
    OP_MULT = 4'b1000,
    OP_RELU = 4'b1001,
    OP_ADD  = 4'b1010;

// Vector source encoding passed to Vector_Processor
parameter [2:0]
    VSRC_MEM     = 3'd0,   // Device memory
    VSRC_SA_OUT  = 3'd1,   // Systolic array output registers
    VSRC_VEC_BUF = 3'd2;   // Vector buffer output

// Vector destination encoding passed to Vector_Processor
parameter [2:0]
    VDST_MEM   = 3'd0,   // Device memory
    VDST_SA_A  = 3'd1,   // Systolic array top input buffers
    VDST_SA_B  = 3'd2,   // Systolic array left input buffers
    VDST_ACT   = 3'd3,   // Activator input
    VDST_ALU_A = 3'd4,   // ALU input A
    VDST_ALU_B = 3'd5;   // ALU input B

// Function codes forwarded to Activator and ALU (from ISA funct3 field)
parameter [2:0]
    FUNCT_RELU = 3'd0,   // ReLU activation (funct3=0x0 per ISA)
    FUNCT_ADD  = 3'd0,   // Element-wise add (funct3=0x0 per ISA)
    FUNCT_NOOP = 3'd7;   // No operation (unused funct3 encoding)

// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// Latched copy of the current instruction, stable from EXEC through writeback
reg [127:0] instr_latch;

// State machine registers
reg [3:0] S, NS;

// Combinational field aliases decoded from instr_latch per ISA v0.4 (all
// formats share the 24-bit rs1 field at bits 123-100).
wire [3:0]  instr_opcode = instr_latch[127:124];
wire [23:0] instr_rs1    = instr_latch[123:100];

// MUL-format field aliases (v0.4)
wire [3:0]  mul_sz11 = instr_latch[99:96];   // rows of matrix 1
wire [3:0]  mul_sz12 = instr_latch[95:92];   // cols of matrix 1
wire [23:0] mul_rs2  = instr_latch[91:68];   // source 2 address
wire [3:0]  mul_sz21 = instr_latch[67:64];   // rows of matrix 2
wire [3:0]  mul_sz22 = instr_latch[63:60];   // cols of matrix 2 (= cols of output)
wire [23:0] mul_rd   = instr_latch[59:36];   // destination address
wire [7:0]  mul_scale = instr_latch[35:28];  // M0: dyadic requant multiplier
wire [7:0]  mul_shift = instr_latch[27:20];  // n:  dyadic requant right-shift

// ACT-format field aliases (relu, v0.4)
wire [23:0] act_rd  = instr_latch[99:76];    // destination address
wire [15:0] act_len = instr_latch[75:60];    // number of elements

// ELEM-format field aliases (add, v0.4)
wire [23:0] elem_rs2    = instr_latch[99:76];  // source 2 address
wire [7:0]  elem_sz1    = instr_latch[75:68];  // row count
wire [7:0]  elem_sz2    = instr_latch[67:60];  // col count
wire [23:0] elem_rd     = instr_latch[59:36];  // destination address
// Total elements for add: sz1 * sz2 (at most 255*255 = 65025 fits in 16 bits)
wire [15:0] elem_length = {8'd0, elem_sz1} * {8'd0, elem_sz2};

// True when all execute-phase vector processors required by the current
// instruction have asserted vector_idle HIGH.
// OP_RELU only uses VP_A; OP_MULT and OP_ADD use both VP_A and VP_B.
wire all_exec_vps_done;
assign all_exec_vps_done = (instr_opcode == OP_RELU) ?
                            i_vector_idle_a :
                            (i_vector_idle_a & i_vector_idle_b);

// State machine driver
always @ (posedge i_clk or negedge i_trst)
begin
    if (i_trst == 1'b0)
        S <= START;
    else
        S <= NS;
end

// Next state logic
always @ (*)
begin
    case (S)
        START:
            NS = IDLE;
        IDLE:
            NS = i_controller_start ? CLEAR : IDLE;
        CLEAR:
            // Use i_instruction directly — instr_latch is updated at the
            // posedge that exits CLEAR, so it holds the previous instruction
            // here. The Feeder holds i_instruction stable during controller
            // processing.
            NS = ((i_instruction[127:124] == OP_MULT) ||
                  (i_instruction[127:124] == OP_RELU) ||
                  (i_instruction[127:124] == OP_ADD))
                 ? EXEC_VP_START : DECODE_ERROR;
        EXEC_VP_START:
            NS = EXEC_VP_WAIT;
        EXEC_VP_WAIT:
            // Route to systolic array only for multiply; otherwise go directly
            // to writeback phase
            NS = all_exec_vps_done ?
                 ((instr_opcode == OP_MULT) ? SA_START : WB_VP_START) :
                 EXEC_VP_WAIT;
        SA_START:
            NS = SA_WAIT;
        SA_WAIT:
            NS = i_systolic_array_idle ? WB_VP_START : SA_WAIT;
        WB_VP_START:
            NS = WB_VP_WAIT;
        WB_VP_WAIT:
            // Writeback always uses VP_A only
            NS = i_vector_idle_a ? IDLE : WB_VP_WAIT;
        STATE_ERROR:
            NS = STATE_ERROR;
        DECODE_ERROR:
            NS = DECODE_ERROR;
        default:
            NS = STATE_ERROR;
    endcase
end

// Execution logic — latches the instruction during the CLEAR cycle so that
// combinational decode signals remain stable through the entire execute and
// writeback phases.
always @ (posedge i_clk or negedge i_trst)
begin
    if (i_trst == 1'b0)
    begin
        instr_latch <= 128'd0;
    end
    else
    begin
        case (S)
            CLEAR:
                instr_latch <= i_instruction;
            default:;
        endcase
    end
end

// controller_idle: HIGH when the controller is waiting for a new instruction
assign o_controller_idle = (S == START || S == IDLE);

// clear: HIGH for exactly one clock cycle (CLEAR state) to flush the vector
// buffer (via TPU sclr logic), ALU, Activator, and Systolic_Array.
assign o_clear = (S == CLEAR);

// vector_start_a: pulsed HIGH during EXEC_VP_START and WB_VP_START states
assign o_vector_start_a = (S == EXEC_VP_START) || (S == WB_VP_START);

// vector_start_b: pulsed HIGH during EXEC_VP_START for mult and add only;
// relu does not use VP_B.
assign o_vector_start_b = (S == EXEC_VP_START) &&
                          ((instr_opcode == OP_MULT) || (instr_opcode == OP_ADD));

// systolic_array_start: pulsed HIGH for one cycle during SA_START
assign o_systolic_array_start = (S == SA_START);

// Activator function: relu during relu instruction, no-op otherwise
assign o_activator_funct = (instr_opcode == OP_RELU) ? FUNCT_RELU : FUNCT_NOOP;

// ALU function: add during add instruction, no-op otherwise
assign o_alu_funct = (instr_opcode == OP_ADD) ? FUNCT_ADD : FUNCT_NOOP;

// Combinational decode of Vector_Processor control signals.
// Signals change between execute phase (EXEC_VP_START / EXEC_VP_WAIT) and
// writeback phase (WB_VP_START / WB_VP_WAIT) based on the current state.
always @ (*)
begin
    // Safe defaults — route nothing and address zero
    o_vect_source_a        = VSRC_MEM;
    o_vect_dest_a          = VDST_MEM;
    o_mem_source_address_a = 24'd0;
    o_mem_dest_address_a   = 24'd0;
    o_length_a             = 16'd0;
    o_dim0_a               = 4'd0;
    o_dim1_a               = 4'd0;
    o_scale_a              = 8'd0;   // don't-care unless MUL writeback
    o_shift_a              = 8'd0;

    o_vect_source_b        = VSRC_MEM;
    o_vect_dest_b          = VDST_MEM;
    o_mem_source_address_b = 24'd0;
    o_length_b             = 16'd0;
    o_dim0_b               = 4'd0;
    o_dim1_b               = 4'd0;

    case (instr_opcode)
        OP_MULT:
        begin
            if (S == EXEC_VP_START || S == EXEC_VP_WAIT)
            begin
                // Execute: VP_A loads rs1 (sz11 x sz12) into top SA input buffers
                o_vect_source_a        = VSRC_MEM;
                o_vect_dest_a          = VDST_SA_A;
                o_mem_source_address_a = instr_rs1;
                o_dim0_a               = mul_sz11;
                o_dim1_a               = mul_sz12;
                // Execute: VP_B loads rs2 (sz21 x sz22) into left SA input buffers
                o_vect_source_b        = VSRC_MEM;
                o_vect_dest_b          = VDST_SA_B;
                o_mem_source_address_b = mul_rs2;
                o_dim0_b               = mul_sz21;
                o_dim1_b               = mul_sz22;
            end
            else
            begin
                // Writeback: VP_A reads SA output (sz11 x sz22) and writes to rd,
                // requantizing each element with the instruction's M0/n.
                o_vect_source_a      = VSRC_SA_OUT;
                o_vect_dest_a        = VDST_MEM;
                o_mem_dest_address_a = mul_rd;
                o_dim0_a             = mul_sz11;
                o_dim1_a             = mul_sz22;
                o_scale_a            = mul_scale;
                o_shift_a            = mul_shift;
            end
        end

        OP_RELU:
        begin
            if (S == EXEC_VP_START || S == EXEC_VP_WAIT)
            begin
                // Execute: VP_A streams act_len elements from rs1 to Activator
                o_vect_source_a        = VSRC_MEM;
                o_vect_dest_a          = VDST_ACT;
                o_mem_source_address_a = instr_rs1;
                o_length_a             = act_len;
            end
            else
            begin
                // Writeback: VP_A reads act_len elements from vector buffer to rd
                o_vect_source_a      = VSRC_VEC_BUF;
                o_vect_dest_a        = VDST_MEM;
                o_mem_dest_address_a = act_rd;
                o_length_a           = act_len;
            end
        end

        OP_ADD:
        begin
            if (S == EXEC_VP_START || S == EXEC_VP_WAIT)
            begin
                // Execute: VP_A streams rs1 to ALU_A, VP_B streams rs2 to ALU_B
                o_vect_source_a        = VSRC_MEM;
                o_vect_dest_a          = VDST_ALU_A;
                o_mem_source_address_a = instr_rs1;
                o_length_a             = elem_length;
                o_vect_source_b        = VSRC_MEM;
                o_vect_dest_b          = VDST_ALU_B;
                o_mem_source_address_b = elem_rs2;
                o_length_b             = elem_length;
            end
            else
            begin
                // Writeback: VP_A reads elem_length elements from vector buffer to rd
                o_vect_source_a      = VSRC_VEC_BUF;
                o_vect_dest_a        = VDST_MEM;
                o_mem_dest_address_a = elem_rd;
                o_length_a           = elem_length;
            end
        end

        default:; // DECODE_ERROR handles invalid opcodes; outputs hold defaults
    endcase
end

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
