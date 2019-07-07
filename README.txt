For OpenMP:

- Running:
$ gcc -fopenmp code.c -o out



For CUDA:

- Enable GPU:
# tee /proc/acpi/bbswitch <<< ON

- Disable GPU:
# rmmod nvidia_uvm
# rmmod nvidia
# tee /proc/acpi/bbswitch <<< OFF

- Running:
$ nvcc code.cu -o out

