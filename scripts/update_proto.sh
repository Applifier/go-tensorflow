#!/bin/bash

set -v
set -e

function cloneOrPull() {
    if [ ! -d "$2" ] ; then
        git clone  "$1" "$2"
    else
        pushd "$2"
        git pull "$1"
        popd
    fi
}

cloneOrPull https://github.com/tensorflow/tensorflow.git .tensorflow_repo
cloneOrPull https://github.com/tensorflow/serving.git .serving_repo


rm -rf types/tensorflow_serving
mkdir -p types/tensorflow_serving/

cp .serving_repo/tensorflow_serving/apis/*.proto types/tensorflow_serving/
cp .serving_repo/tensorflow_serving/config/*.proto types/tensorflow_serving/
cp .serving_repo/tensorflow_serving/util/*.proto types/tensorflow_serving/
cp .serving_repo/tensorflow_serving/core/*.proto types/tensorflow_serving/
cp .serving_repo/tensorflow_serving/sources/storage_path/*.proto types/tensorflow_serving/

find types/tensorflow_serving -type f -name '*.proto' -exec sed -i '' 's/tensorflow_serving\/apis/tensorflow_serving/g' {} \;
find types/tensorflow_serving -type f -name '*.proto' -exec sed -i '' 's/tensorflow_serving\/config/tensorflow_serving/g' {} \;
find types/tensorflow_serving -type f -name '*.proto' -exec sed -i '' 's/tensorflow_serving\/util/tensorflow_serving/g' {} \;
find types/tensorflow_serving -type f -name '*.proto' -exec sed -i '' 's/tensorflow_serving\/core/tensorflow_serving/g' {} \;
find types/tensorflow_serving -type f -name '*.proto' -exec sed -i '' 's/tensorflow_serving\/sources\/storage_path/tensorflow_serving/g' {} \;

rm -rf types/tensorflow

mkdir -p types/tensorflow/core/framework
mkdir -p types/tensorflow/core/example
mkdir -p types/tensorflow/core/lib/core
mkdir -p types/tensorflow/core/protobuf


cp .tensorflow_repo/tensorflow/core/framework/*.proto  types/tensorflow/core/framework/
cp .tensorflow_repo/tensorflow/core/example/*.proto types/tensorflow/core/example/
cp .tensorflow_repo/tensorflow/core/lib/core/*.proto types/tensorflow/core/lib/core/
cp .tensorflow_repo/tensorflow/core/protobuf/{saver,meta_graph,config,named_tensor,debug,cluster,rewriter_config,saved_model}.proto types/tensorflow/core/protobuf/

# option go_package = "github.com/tensorflow/tensorflow/tensorflow/go -> 
find types/tensorflow -type f -name '*.proto' -exec sed -i '' 's/github.com\/tensorflow\/tensorflow\/tensorflow\/go/github.com\/Applifier\/go-tensorflow\/types\/tensorflow/g' {} \;


PROTOC_OPTS='-I types --gogofaster_out=plugins=grpc,paths=source_relative,\
Mgoogle/protobuf/any.proto=github.com/gogo/protobuf/types,\
Mgoogle/protobuf/duration.proto=github.com/gogo/protobuf/types,\
Mgoogle/protobuf/struct.proto=github.com/gogo/protobuf/types,\
Mgoogle/protobuf/timestamp.proto=github.com/gogo/protobuf/types,\
Mgoogle/protobuf/wrappers.proto=github.com/gogo/protobuf/types:types'

eval "protoc $PROTOC_OPTS types/tensorflow_serving/*.proto"
eval "protoc $PROTOC_OPTS types/tensorflow/core/framework/*.proto"
eval "protoc $PROTOC_OPTS types/tensorflow/core/example/*.proto"
eval "protoc $PROTOC_OPTS types/tensorflow/core/lib/core/*.proto"
eval "protoc $PROTOC_OPTS types/tensorflow/core/protobuf/*.proto"

cp -a types/github.com/Applifier/go-tensorflow/types/tensorflow/* types/tensorflow
rm -Rf types/github.com