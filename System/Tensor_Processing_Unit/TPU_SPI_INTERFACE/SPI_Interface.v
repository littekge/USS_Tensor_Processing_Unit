/*
Module to interface with the SPI connection.
The SPI_Slave module is connected to a FIFO buffer. Upper level modules can
then read from the buffer and used the data that was transferred.

*/

module SPI_Interface(
	//clock and reset
	input clk,
	input rst,
	
	//buffer interfacing signals
	input i_rdreq,
	input i_sclr,
	output o_empty,
	output o_full,
	output [7:0] o_q,
	output [7:0] o_usedw,
	
	//SPI Signals
	input i_SPI_Clk,
	output o_SPI_MISO,
	input i_SPI_MOSI,
	input i_SPI_SS
);

SPI_Input_Buffer inbuf (
	.clock(clk),
	.data(recieve_Byte),
	.rdreq(i_rdreq),
	.sclr(i_sclr),
	.wrreq(recieve_DV),
	.empty(o_empty),
	.full(o_full),
	.q(o_q),
	.usedw(o_usedw)
);

//interfacing wires
wire recieve_DV;
wire [7:0] recieve_Byte;

SPI_Slave #(SPI_MODE = 0) slave1 (
	.i_Rst_L(rst), //FPGA reset
	.i_Clk(clk), //FPGA clock
	
	.o_RX_DV(recieve_DV), //SPI recieve data valid
	.o_RX_Byte(recieve_Byte), //SPI recieve data
	.i_TX_DV(1'b0), //SPI send data valid
	.i_TX_Byte(8'h00), //SPI send data
	
	//SPI Signals
	.i_SPI_Clk(i_SPI_Clk),
	.o_SPI_MISO(o_SPI_MISO),
	.i_SPI_MOSI(i_SPI_MOSI),
	.i_SPI_CS_n(i_SPI_SS)
);

endmodule