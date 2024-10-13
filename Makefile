build:
	docker build . -t bash-clip
	docker volume create bash_clip_cache

run: build
	docker run -it --rm -v clip_cache:/root/.cache -v $(PWD):/workspace clip /bin/bash
