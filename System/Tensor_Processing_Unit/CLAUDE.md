# Claude - Project Instructions

> These rules apply to every task in this project.
> Claude reads this file to begin the coding process.

> **NOTE:** All paths are described relative to the directory in which this file is located unless otherwise specified.

## Project Context

- **Target Platform:** Terasic DE1-SoC FPGA
- **Language:** Verilog HDL 2001
- **Development Environment:** Quartus Prime Lite Edition
- **Development Platform:** Windows 11
- **Build Spec**: `main.md` — the single source of truth for what to build (includes 6-step build plan).
- **Change Log**: `log.md` — append an entry every time you make a change.
- **Specifications:** — located at `../../Specifications/`
	- **Instruction Set Architecture:** `Functional_TPU_ISA_v0.2.md` (TPU instruction set)
	- **Messaging Protocol:** `Functional_TPU_Message_Protocol_v0.1.md` (defines byte format for communicating with the TPU)
	- **Hardware Spec:** `Functional_TPU_Hardware_Specification_v0.1.1.md` (detailed description of Verilog module implementations)

## Iterative Build Workflow

1. **Before starting work**, read `main.md` to understand the build spec and `log.md` to see what's already been done.
2. **Work in small increments** — one module or feature at a time.
3. **After every change**, append a dated entry to `log.md` describing what was added, changed, or fixed.
4. **Review code** after every build step, identifying and correcting syntax and logic errors from the initial build.
5. **Generate tests** after every build step. Build Verilog testbenches and instructions on how to run them in the `/tests` folder, including what it looks like for each test to be successful and unsuccessful (test cases should evaluate the implementation of the most recent build step, so there should be 7 in total). **Testbenches should NOT synthesize away any signals and log every signal waveform by default.** An example vsim command that accomplishes this is shown below.
`vsim -L altera_mf_ver -voptargs=+acc -gui work.TB_Step1_Programmer -do "add wave -r /*; run -all"`
6. **Commit often** with descriptive messages: `git add . ; git commit -m "feat: ..."`. Stay on the same git branch for the duration of the build.
7. **Mark completed build steps** in `main.md`.

## Coding Practices for Verilog Files

### Module Definitions and Headers
- Each .v file must declare and fully contain only a single module.
- Input and outputs as well as net type should be declared in the module header whenever possible.
- All input names must have an i_ prefix.
- All output names must have an o_ prefix.

### Code Structure
- Module code should be structured into parameter, main code, and debug sections as follows: 

```verilog
/* 
 * File: 
 * Author: 
 * Date: 
 * 
 * Insert module description here
 */
module moduleName (

    //Insert regular inputs and outputs here.

    // ---------- PARAMETERS ---------- //
	//Parameters to pass through.
	// ---------- END PARAMETERS ---------- //

	// ---------- DEBUG ---------- //
	//Debug inputs and outputs, usually switches, buttons, and LEDs.
	// ---------- END DEBUG ---------- //
);

// ---------- PARAMETERS ---------- //
//Insert parameter declarations here.
// ---------- END PARAMETERS ---------- //

// ---------- CODE ---------- //
//Insert module code here.
// ---------- END CODE ---------- //

// ---------- DEBUG ---------- //
//Insert debug code here.
// ---------- END DEBUG ---------- //
endmodule
```

- Standard comments and block comments should be used for further subheadings or labels.
- The description section should not exceed column 80, instead continuing onto a new line.

### State Machines
- State machines define a current state and next state register.
- State machines define available states as Verilog parameters.
- State machines are built using three Verilog always blocks:
	- Small sequential block to drive the state machine (assigns next state to current state every clock cycle)
	- Combinational block that determines the next state
	- Sequential block to determine execution logic (what sequential actions occur during each state)
- State machines **always** have an initial START state that has no execution logic.

An example of a properly structured state machine is given below. This state machine writes ascii characters to memory.

```verilog
//state machine states
parameter 
	ERROR = 3'd0,
	START = 3'd1,
	WAIT_WREN = 3'd2,
	COL_CHECK = 3'd3,
	WRITE = 3'd4,
	COL_INC = 3'd5,
	ROW_INC = 3'd6;

reg [3:0] col_num; //column number
reg [7:0] row_num; //row number

reg [31:0] data; //register to store input data

//State machine variables
reg [2:0]S, NS;

//State machine driver
always @ (posedge i_clk or negedge i_rst)
begin
	if (i_rst == 1'b0)
	begin
		S <= START; //sets state to START if reset is triggered
	end
	else
	begin
		S <= NS; //otherwise set S to NS
	end
end

//Determining next state
always @ (*)
begin
	case (S)
		START: NS = WAIT_WREN;
		WAIT_WREN: NS = (i_write_next == 1'b1)?(COL_CHECK):(WAIT_WREN);  //idles in WAIT_WREN until write next signal is recieved
		COL_CHECK: NS = (col_num < 4'd9)?(WRITE):(ROW_INC); //operates a for loop 
		WRITE: NS = COL_INC;
		COL_INC: NS = COL_CHECK;
		ROW_INC: NS = WAIT_WREN;
		default: NS = ERROR;
	endcase
end

//Execution logic
always @ (posedge i_clk or negedge i_rst)
begin
	if (i_rst == 1'b0)
	begin
		//zeroing registers on reset
		row_num <= 8'd0;
		col_num <= 4'd0;
		ascii_wren <= 1'd0;
		o_write_ready <= 1'd0;
		data <= 32'd0;
	end
	else
	begin
		case (S)
			WAIT_WREN:
			begin
				o_write_ready <= 1'b1; //indicates waiting for next input
				data <= i_data;  //save current data (avoids strange behavior if input is changed asynchronously)
			end
			COL_CHECK:
			begin
				o_write_ready <= 1'b0; //indicates operation
				col_num <= (NS == ROW_INC)?(4'd0):(col_num); //resetting column number when exiting for loop
			end
			WRITE:
			begin
				ascii_wren <= 1'b1; //write to ASCII memory
			end
			COL_INC:
			begin
				ascii_wren <= 1'b0;
				col_num <= col_num + 4'd1; //increment column number
			end
			ROW_INC:
			begin
				row_num <= row_num + 8'd1; //increment row number
			end
			default:; //do nothing
		endcase
	end
end
```

### Comment Rules

Follow three rules for comments:

- **Rule 1 — Names explain *what***: Choose clear, descriptive names for classes, methods, and variables. If the name is good enough, no comment is needed to explain what it does.
- **Rule 2 — Code explains *how***: The code itself should be readable enough to show how things work. Don't write comments that restate the code.
- **Rule 3 — Comments explain *why***: Only add comments when the reason behind a decision isn't obvious from the code. Explain *why* this approach was chosen, *why* a workaround exists, or *why* a non-obvious value is used.

### Other Conventions
- **Named Instantiation:** When instantiating a Verilog module connect inputs and outputs explicitly using named instantiation.
- **Nested Ternary Operators:** Nested ternary operators are not encouraged, but permissible if absolutely necessary.
- **Code Simplicity:** Code should be simple and readable. No single line of code should implement an excessive amount of operations (combinational logic can be especially difficult to read if implemented on one line).
- **DO NOT** modify the debug sections of modules. I will edit those sections and use them for debug signals when synthesizing to the FPGA hardware.

## Architecture Rules

- **Predefined Architecture:** All files in the `/TPU` subdirectory are already defined. Do not generate any new Verilog (.v) files.
- **Markdown File Modification:** Do not modify existing markdown files unless explicitly specified by another instruction. Do not generate any new markdown files. All changes can be sufficiently recorded in `log.md`

## Log Format

When appending to `log.md`, use this format:

```markdown
## YYYY-MM-DD — Short Title

- What was done (bullet points)
- Files created or modified
- Tests added or updated
- Any issues encountered
```

## Chat Responses
- **Summarize** responses into brief, high-level overviews to optimize token usage.
- **Remove** extra conversational prose (introductory text, pleasantries, etc.). 
- **Prioritize** functional, bulleted responses.
