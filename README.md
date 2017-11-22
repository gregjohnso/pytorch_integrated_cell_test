## pytorch_integrated_cell_test

test for different versions of pytorch

runme.sh - runs the model with dummy data

run_docker_pyt3.sh - runs runme.sh from a dockerfile built with the Pytorch v0.3 branch
run_docker_master.sh - runs runme.sh from a dockerfile built with the Pytorch Master branch as of approximately 4:00 pm PST on 21/11/2017 

This is intentended to be run on a machine with 3 Pascal Titan Xs

### Observations
	run_docker_pyt3.sh - runs fine
	run_docker_master.sh - runs out of memory
