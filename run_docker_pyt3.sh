nvidia-docker run \
	-it \
	-v /allen/aics/modeling/gregj/projects:/root/projects \
	pytorch/v0.3.0:latest bash -c "cd /root/projects/pytorch_integrated_cell_lite; bash runme.sh"
