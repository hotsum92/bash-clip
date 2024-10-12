build:
	docker build . -t bash-clip
	docker volume create bash_clip_cache
