export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

alexandre:
	python src/im2meas.py img/al/*

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

run:
	python src/im2meas.py img/*.*
run_calibration:
	python src/im2meas.py img/*.* --calibration 1
run_with_mass:
	python src/im2meas.py img/* -m "${mass}"

biomake:
	python src/biomake/biomake_models.py --bioModOptions src/biomake/tech_opt.yml "${name}.txt" > "${name}.bioMod"

bioviz:
	python src/biov.py "${name}"
