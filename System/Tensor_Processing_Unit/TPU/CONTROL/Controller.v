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

    // Active leading dimension (physical row stride) supplied to each vector
    // processor: ld1 when loading src1, ld2 when loading src2, ld2 at writeback.
    output reg  [23:0] o_leading_dimension_a,
    output reg  [23:0] o_leading_dimension_b,

    // Activator and ALU function selects
    output wire [2:0]  o_activator_funct,
    output wire [2:0]  o_alu_funct,

    // Systolic_Array control outputs
    output wire        o_systolic_array_start,
    // Pulsed HIGH for one cycle after writeback to zero the SA accumulators.
    // Asserted only when funct3 selects clearing (mult, not multip).
    output wire        o_accumulator_clear,

    // Pooler function select (max during a max instruction, NO-OP otherwise)
    output wire [2:0]  o_pooler_funct,

    // Window descriptor registers (WINCONFIG). Supplied to the vector
    // processors for the windowed instructions (im2col / max); persist until
    // the next window instruction or trst.
    output wire [11:0] o_win_chans,
    output wire [11:0] o_win_inh,
    output wire [11:0] o_win_inw,
    output wire [11:0] o_win_outh,
    output wire [11:0] o_win_outw,
    output wire [3:0]  o_win_winh,
    output wire [3:0]  o_win_winw,
    output wire [3:0]  o_win_strh,
    output wire [3:0]  o_win_strw,
    output wire [3:0]  o_win_padh,
    output wire [3:0]  o_win_padw

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
    DECODE_ERROR  = 4'd10,
    ACC_CLEAR     = 4'd11;   // one-cycle accumulator-clear phase after writeback

// ISA instruction opcodes
parameter [3:0]
    OP_MULT   = 4'b1000,   // shared by mult / multip, distinguished by funct3
    OP_RELU   = 4'b1001,
    OP_ADD    = 4'b1010,
    OP_MAX    = 4'b1100,   // POOL    (max pooling)
    OP_IM2COL = 4'b1101,   // SHAPE   (im2col gather)
    OP_WINDOW = 4'b1110,   // WINCONFIG (load window descriptor)
    OP_STRIDE = 4'b1111;   // LDCONFIG  (load leading dimensions)

// funct3 sub-codes for the MUL opcode (per ISA Instruction Set Listing)
parameter [2:0]
    FUNCT3_MULT   = 3'h0,   // clears the accumulator after writeback
    FUNCT3_MULTIP = 3'h1;   // leaves the accumulator to keep accumulating

// Vector source encoding passed to Vector_Processor
parameter [2:0]
    VSRC_MEM     = 3'd0,   // Device memory
    VSRC_SA_OUT  = 3'd1,   // Systolic array output registers
    VSRC_VEC_BUF = 3'd2,   // Vector buffer output
    VSRC_MEM_WIN = 3'd3;   // Device memory, windowed gather (im2col / max)

// Vector destination encoding passed to Vector_Processor
parameter [2:0]
    VDST_MEM   = 3'd0,   // Device memory
    VDST_SA_A  = 3'd1,   // Systolic array top input buffers
    VDST_SA_B  = 3'd2,   // Systolic array left input buffers
    VDST_ACT   = 3'd3,   // Activator input
    VDST_ALU_A = 3'd4,   // ALU input A
    VDST_ALU_B = 3'd5,   // ALU input B
    VDST_POOL  = 3'd6;   // Pooler input

// Function codes forwarded to Activator, ALU, and Pooler (ISA funct3 field)
parameter [2:0]
    FUNCT_RELU = 3'd0,   // ReLU activation (funct3=0x0 per ISA)
    FUNCT_ADD  = 3'd0,   // Element-wise add (funct3=0x0 per ISA)
    FUNCT_MAX  = 3'd0,   // Max pooling (funct3=0x0 per ISA)
    FUNCT_NOOP = 3'd7;   // No operation (unused funct3 encoding)

// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //

// Latched copy of the current instruction, stable from EXEC through writeback
reg [127:0] instr_latch;

// State machine registers
reg [3:0] S, NS;

// Combinational field aliases decoded from instr_latch per ISA v0.5 (all
// formats share the 24-bit rs1 field at bits 123-100).
wire [3:0]  instr_opcode = instr_latch[127:124];
wire [2:0]  instr_funct3 = instr_latch[2:0];
wire [23:0] instr_rs1    = instr_latch[123:100];

// Leading-dimension registers (ISA v0.5). Loaded by the stride instruction,
// held until the next stride or trst, and cleared to 0 (contiguous) on trst.
reg [23:0] ld1;
reg [23:0] ld2;

// CONFIG-format field aliases (stride): im1 -> ld1, im2 -> ld2.
wire [23:0] cfg_im1 = i_instruction[123:100];
wire [23:0] cfg_im2 = i_instruction[99:76];

// Window descriptor registers (ISA v0.6). Loaded by the window (WINCONFIG)
// instruction and held until the next window or trst. No default value; a
// window must be issued before any windowed instruction (im2col / max).
reg [11:0] chans, inh, inw, outh, outw;
reg [3:0]  winh, winw, strh, strw, padh, padw;

// WINCONFIG (W-format) field aliases, decoded from i_instruction directly
// because (like stride) the descriptor is latched at the posedge that exits
// CLEAR, when instr_latch still holds the previous instruction.
wire [11:0] win_inh   = i_instruction[123:112];
wire [11:0] win_inw   = i_instruction[111:100];
wire [11:0] win_chans = i_instruction[99:88];
wire [11:0] win_outh  = i_instruction[87:76];
wire [11:0] win_outw  = i_instruction[75:64];
wire [3:0]  win_winh  = i_instruction[63:60];
wire [3:0]  win_winw  = i_instruction[59:56];
wire [3:0]  win_strh  = i_instruction[55:52];
wire [3:0]  win_strw  = i_instruction[51:48];
wire [3:0]  win_padh  = i_instruction[47:44];
wire [3:0]  win_padw  = i_instruction[43:40];

// SHAPE / POOL (A-format) field aliases for im2col / max: src1 = instr_rs1
// (bits 123-100, shared alias), dest at bits 99-76.
wire [23:0] shape_rd = instr_latch[99:76];

// Writeback element count for max: chans * outh * outw pooled results are
// streamed to the vector buffer during Execute and copied to dest in Writeback.
wire [15:0] max_out_len = chans * outh * outw;

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
// OP_RELU, OP_IM2COL, and OP_MAX only use VP_A; OP_MULT and OP_ADD use both.
wire single_vp_op = (instr_opcode == OP_RELU)   ||
                    (instr_opcode == OP_IM2COL) ||
                    (instr_opcode == OP_MAX);
wire all_exec_vps_done;
assign all_exec_vps_done = single_vp_op ?
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
            // A register-configuration instruction (stride/window) only latches
            // its fields (done in the execution block) and has no Execute or
            // Writeback phase, so it returns straight to IDLE.
            if ((i_instruction[127:124] == OP_STRIDE) ||
                (i_instruction[127:124] == OP_WINDOW))
                NS = IDLE;
            else if ((i_instruction[127:124] == OP_MULT)   ||
                     (i_instruction[127:124] == OP_RELU)   ||
                     (i_instruction[127:124] == OP_ADD)    ||
                     (i_instruction[127:124] == OP_IM2COL) ||
                     (i_instruction[127:124] == OP_MAX))
                NS = EXEC_VP_START;
            else
                NS = DECODE_ERROR;
        EXEC_VP_START:
            NS = EXEC_VP_WAIT;
        EXEC_VP_WAIT:
            // Multiply routes through the systolic array; im2col writes its
            // result directly to memory during Execute so it skips Writeback and
            // returns to IDLE; all others (relu / add / max) go to Writeback.
            NS = all_exec_vps_done ?
                 ((instr_opcode == OP_MULT)   ? SA_START :
                  (instr_opcode == OP_IM2COL) ? IDLE     :
                  WB_VP_START) :
                 EXEC_VP_WAIT;
        SA_START:
            NS = SA_WAIT;
        SA_WAIT:
            NS = i_systolic_array_idle ? WB_VP_START : SA_WAIT;
        WB_VP_START:
            NS = WB_VP_WAIT;
        WB_VP_WAIT:
            // Writeback always uses VP_A only
            NS = i_vector_idle_a ? ACC_CLEAR : WB_VP_WAIT;
        ACC_CLEAR:
            // One cycle after writeback; accumulator_clear is asserted here
            // only when funct3 selects clearing (mult, not multip).
            NS = IDLE;
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
        ld1         <= 24'd0;
        ld2         <= 24'd0;
        // Window descriptor has no default value per the ISA, but is cleared on
        // trst so a windowed instruction issued without a preceding window sees
        // a defined (zero) descriptor rather than stale/undefined values.
        chans       <= 12'd0;
        inh         <= 12'd0;
        inw         <= 12'd0;
        outh        <= 12'd0;
        outw        <= 12'd0;
        winh        <= 4'd0;
        winw        <= 4'd0;
        strh        <= 4'd0;
        strw        <= 4'd0;
        padh        <= 4'd0;
        padw        <= 4'd0;
    end
    else
    begin
        case (S)
            CLEAR:
            begin
                instr_latch <= i_instruction;
                // Register-configuration instructions latch their fields here.
                // Uses i_instruction (same timing as the instr_latch load).
                if (i_instruction[127:124] == OP_STRIDE)
                begin
                    ld1 <= cfg_im1;
                    ld2 <= cfg_im2;
                end
                if (i_instruction[127:124] == OP_WINDOW)
                begin
                    chans <= win_chans;
                    inh   <= win_inh;
                    inw   <= win_inw;
                    outh  <= win_outh;
                    outw  <= win_outw;
                    winh  <= win_winh;
                    winw  <= win_winw;
                    strh  <= win_strh;
                    strw  <= win_strw;
                    padh  <= win_padh;
                    padw  <= win_padw;
                end
            end
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

// accumulator_clear: pulsed HIGH for one cycle in ACC_CLEAR (after writeback)
// only when the decoded MUL instruction's funct3 selects clearing (mult, not
// multip). This lets a multip run keep accumulating in-array while a
// terminating mult zeroes the accumulator once its result is written back.
assign o_accumulator_clear = (S == ACC_CLEAR) &&
                             (instr_opcode == OP_MULT) &&
                             (instr_funct3 == FUNCT3_MULT);

// Activator function: relu during relu instruction, no-op otherwise
assign o_activator_funct = (instr_opcode == OP_RELU) ? FUNCT_RELU : FUNCT_NOOP;

// ALU function: add during add instruction, no-op otherwise
assign o_alu_funct = (instr_opcode == OP_ADD) ? FUNCT_ADD : FUNCT_NOOP;

// Pooler function: max during a max instruction, no-op otherwise
assign o_pooler_funct = (instr_opcode == OP_MAX) ? FUNCT_MAX : FUNCT_NOOP;

// Window descriptor outputs mirror the persistent descriptor registers so the
// vector processors always see the descriptor set by the most recent window.
assign o_win_chans = chans;
assign o_win_inh   = inh;
assign o_win_inw   = inw;
assign o_win_outh  = outh;
assign o_win_outw  = outw;
assign o_win_winh  = winh;
assign o_win_winw  = winw;
assign o_win_strh  = strh;
assign o_win_strw  = strw;
assign o_win_padh  = padh;
assign o_win_padw  = padw;

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

    // Default contiguous (0) unless a MUL supplies a leading dimension below.
    o_leading_dimension_a  = 24'd0;
    o_leading_dimension_b  = 24'd0;

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
                // src1 is walked with ld1, src2 with ld2 (0 = contiguous).
                o_leading_dimension_a  = ld1;
                o_leading_dimension_b  = ld2;
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
                // dest is written with ld2 (0 = contiguous).
                o_leading_dimension_a = ld2;
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

        OP_IM2COL:
        begin
            // im2col has only an Execute phase: VP_A gathers the windowed
            // feature map at src1 and writes the dense im2col matrix directly to
            // dest. The window descriptor (routed via the o_win_* outputs) sets
            // the element count and iteration order, so length/dim are unused.
            o_vect_source_a        = VSRC_MEM_WIN;
            o_vect_dest_a          = VDST_MEM;
            o_mem_source_address_a = instr_rs1;
            o_mem_dest_address_a   = shape_rd;
        end

        OP_MAX:
        begin
            if (S == EXEC_VP_START || S == EXEC_VP_WAIT)
            begin
                // Execute: VP_A streams each channel's window to the Pooler,
                // asserting window_end on each window's final element. The
                // Pooler writes one pooled value per window to the vector buffer.
                o_vect_source_a        = VSRC_MEM_WIN;
                o_vect_dest_a          = VDST_POOL;
                o_mem_source_address_a = instr_rs1;
            end
            else
            begin
                // Writeback: VP_A copies the chans*outh*outw pooled results from
                // the vector buffer to dest (linear, like relu/add writeback).
                o_vect_source_a      = VSRC_VEC_BUF;
                o_vect_dest_a        = VDST_MEM;
                o_mem_dest_address_a = shape_rd;
                o_length_a           = max_out_len;
            end
        end

        default:; // DECODE_ERROR handles invalid opcodes; outputs hold defaults
    endcase
end

// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
// ---------- END DEBUG ---------- //
endmodule
