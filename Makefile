MODELS_PATH := `pwd`/testdata/models

update-proto:
	./scripts/update_proto.sh
.PHONY: update-proto

run-serving:
	(cd tools/docker && docker-compose up serving)
.PHONY: run-serving

test:
	(cd tools/docker && docker-compose run test)
	(cd tools/docker && docker-compose down)
.PHONY: test
