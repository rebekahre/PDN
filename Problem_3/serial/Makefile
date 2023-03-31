run:
	make all
	./convolution_serial 2048 2048 /home/lee0305/test_data/Problem_3/mat_input.csv mat_output.csv time.csv
	

all: convolution_serial.c
	gcc -g -Wall -o convolution_serial convolution_serial.c

clean:
	$(RM) convolution_serial