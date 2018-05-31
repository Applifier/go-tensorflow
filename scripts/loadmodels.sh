#!/bin/bash

mkdir -p testdata/models/mobilenet/1
cd testdata/models/mobilenet/1
curl http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz | tar xvz  --strip-components=2 ssd_mobilenet_v1_coco_2017_11_17/saved_model