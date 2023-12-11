export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
alexandre:
	python src/im2meas.py img/a/*

william:
	python src/im2meas.py img/william/w/*

run:
	python src/im2meas.py img/*

biomake:
	python src/biomake/biomake_models.py --bioModOptions src/biomake/tech_opt.yml "${name}.txt" > "${name}.bioMod"
bioviz:
	python src/biov.py "${name}"
