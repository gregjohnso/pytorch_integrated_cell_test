## pytorch_integrated_cell_test

test for different versions of pytorch  

**runme.sh** - runs the model with dummy data  

**run_docker_pyt3.sh** - runs runme.sh from a dockerfile built with the Pytorch v0.3 branch  
**run_docker_master.sh** - runs runme.sh from a dockerfile built with the Pytorch Master branch as of approximately 4:00 pm PST on 21/11/2017 

This is intended to be run on a machine with 3 Pascal Titan Xs

### Observations
**run_docker_pyt3.sh** - runs fine  
**run_docker_master.sh** - runs out of memory  


### Notes
in **runme.sh** change GPU IDs and batch size according to system specs

The training occurs on the top half of the model presented here: https://arxiv.org/pdf/1705.00092.pdf
