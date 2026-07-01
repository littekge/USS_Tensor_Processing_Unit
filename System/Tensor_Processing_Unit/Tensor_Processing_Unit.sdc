create_clock -name "CLOCK_50" -period 20.000ns [get_ports {CLOCK_50}]
create_generated_clock -name "CLOCK_25" -divide_by 2 -source \
	[get_ports {CLOCK_50}] [get_pins {debug:dbg|ascii_master_controller:controller|vga_controller:controller|clock_divider:clock|clk_25|q}]
create_clock -name spi_clk -period 8us [get_ports {GPIO_0[7]}]
derive_clock_uncertainty