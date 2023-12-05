export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
run_alexandre:
	python src/im2meas.py img/alexandre_front.jpg img/alexandre_side.jpg img/alexandre_pike.jpg img/alexandre_r_pike.jpg

run_william:
	python src/im2meas.py img/william/william_front.jpg img/william/william_side.jpg img/william/william_pike.jpg img/william/william_r_pike.jpg
