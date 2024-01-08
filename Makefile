export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

alexandre:
	python src/im2meas.py img/a/*

kael:
	python src/im2meas.py img/kael/k/*

william:
	python src/im2meas.py img/william/w/w/*

martin:
	python src/im2meas.py img/m/*

run:
	python src/im2meas.py img/*

run_with_mass:
	python src/im2meas.py img/* -m "${mass}"

biomake:
	python src/biomake/biomake_models.py --bioModOptions src/biomake/tech_opt.yml "${name}.txt" > "${name}.bioMod"

bioviz:
	python src/biov.py "${name}"
