export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

run:
	python src/im2meas.py img/*.*

run_calibration:
	python src/im2meas.py img/*.* --calibration 1
run_luminosity:
	python src/im2meas.py img/*.* -l 1
run_rotate:
	python src/im2meas.py img/*.* --rotation 1
run_with_mass:
	python src/im2meas.py img/* -m "${mass}"
biomake:
	python src/biomake/biomake_models.py --bioModOptions src/biomake/tech_opt.yml "${name}.txt" > "${name}.bioMod"

bioviz:
	python src/biov.py "${name}"

comparison:
	python src/comparison.py "${meas}".bioMod "${gen}".bioMod



#for debug:
# To run the code with more options than just 1 everytime you have to export your pythonpath use:
# YOUR_PATH=pwd
# export PYTHONPATH=YOUR_PATH
#
alexandre:
	python src/im2meas.py img/al/*

romane:
	python src/im2meas.py img/romane/*
josee:
	python src/im2meas.py img/josee/*
mathieu:
	python src/im2meas.py img/mathieu/*
francisca:
	python src/im2meas.py img/francisca/*

kael:
	python src/im2meas.py img/kl/*

eve:
	python src/im2meas.py img/e/*

pierre:
	python src/im2meas.py img/pierre/*

william:
	python src/im2meas.py img/william/w/w/*

william_calib:
	python src/im2meas.py img/william/w/w/* --calibration 1

martin:
	python src/im2meas.py img/m/*

mohammad:
	python src/im2meas.py img/mohammad/*
