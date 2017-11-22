rm ./output/data.pyt

python ./train_model.py \
		--gpu_ids 0 1 2 \
		--save_dir ./output/ \
		--model_name my_model_half \
		--lrEnc 5E-5 --lrDec 5E-5 --lrEncD 1E-2 --lrDecD 5E-5 \
		--encDRatio 1E-2 --decDRatio 1E-6 \
		--noise 1E-2 \
		--nlatentdim 128 \
		--batch_size 30 \
		--nepochs 100 --nepochs_pt2 125 \
		--train_module aaegan_trainv6 \
		--imdir ./output/dummy \
		--dataProvider DataProviderDummy_half \
		--saveStateIter 1 --saveProgressIter 1 \
		--channels_pt1 0 2 --channels_pt2 0 1 2 \
		--ndat 1000
