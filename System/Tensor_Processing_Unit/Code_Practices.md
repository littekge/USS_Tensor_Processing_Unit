# Coding Practices for Verilog Files
This is the commenting syntax and coding practices for verilog files and development.

## Modules
1. Each .v file must declare and fully contain only a single module.
2. Input and outputs as well as net type should be declared in the module header whenever possible.
3. All inputs must have an i_ prefix.
4. All outputs must have an o_ prefix.
5. All inputs and outputs must have a comment describing their function.

## Code
1. Module code should be structured by heading as follows:

```
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

2. Standard comments and block comments should be used for further subheadings or labels.
3. Module descriptions should not exceed column 80.