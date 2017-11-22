nvidia-docker run \
	-it \
	-v /allen/aics/modeling/gregj/projects:/root/projects \
	pytorch:latest bash -c "cd /root/projects/pytorch_integrated_cell_lite; bash runme.sh"
