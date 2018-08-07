MODELS_PATH := `pwd`/testdata/models

build: load-models generate-models
.PHONY: build

generate-models:
	./scripts/generatemodels.sh
.PHONY: generate-models

update-proto:
	./scripts/update_proto.sh
.PHONY: update-proto

run-serving:
	(cd tools/docker && docker-compose up serving)
.PHONY: run-serving

load-models:
	./scripts/loadmodels.sh
.PHONY: load-models

test:
	(cd tools/docker && docker-compose run test)
	(cd tools/docker && docker-compose down)
.PHONY: test

run-mobilenet:
	(cd tools/docker && docker-compose run mobilenet)
	(cd tools/docker && docker-compose down)
.PHONY: run-mobilenet