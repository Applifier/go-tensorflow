#!/bin/bash

docker run --rm -it -v "$GOPATH":/go -w /go/src/github.com/Applifier/go-tensorflow/scripts/models/ tensorflow/tensorflow:1.8.0-py3 sh -c '
rm -Rf ../../testdata/test_models
for filename in *.py; do
    echo "Running $filename"
    python3 $filename
done
'