export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
alexandre:
	python src/im2meas.py img/a/*

william:
	python src/im2meas.py img/william/w/*
