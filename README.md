# CARTS

### Dependencies

- LLVM [v18]/MLIR Dependencies
	- Ninja
	- Gcc greater than 11.4
	- Cmake greater than 3
- ARTS Dependencies
	- librdma dependency* 
- Uses modified versions of Polygeist, LLVM/MLIR, and ARTS
	- All modified versions will be in the Gitlab

### Building the code

- Obtain the code (External PNNL GitHub)
	- git clone https://github.com/pnnl/CARTS
	- cd compiler
- Run the Makefile file under the carts folder:
	- $ make polygeist-download 
	- $ make llvm
	- $ make polygeist
	- $ make arts-download
	- $ make arts
	- $ make build
	- $ make enable
	- $ source enable
-Set the LD_LIBRARY_PATH to point to the ARTS library
	-${src}/.install/arts/lib/libarts.so

### Eample runs

- Go to ${src}/examples/taskwithdeps/for
- Ensure that the arts.cfg is present in the folder
	- Runtime definition for ARTS
- Run the Makefile in that folder and then run the resulting binary 

