
Tool for testing a saved model

```sh
$ go run main.go --modelpath ../../testdata/models/wide_deep/1527087570 -input '{"inputs": {"age":35,"capital_gain":0,"capital_loss":0,"education":"Masters","education_num":14,"gender":"Female","hours_per_week":29,"native_country":"United-States","occupation":"Prof-specialty","relationship":"Husband","workclass":"Private"}}'
Input: map[inputs:map[occupation:Prof-specialty relationship:Husband workclass:Private education:Masters education_num:14 hours_per_week:29 native_country:United-States age:35 capital_gain:0 capital_loss:0 gender:Female]]
2018-09-04 11:44:26.398483: I tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: ../../testdata/models/wide_deep/1527087570
2018-09-04 11:44:26.404779: I tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2018-09-04 11:44:26.414817: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
2018-09-04 11:44:26.434207: I tensorflow/cc/saved_model/loader.cc:113] Restoring SavedModel bundle.
2018-09-04 11:44:26.456225: I tensorflow/cc/saved_model/loader.cc:148] Running LegacyInitOp on SavedModel bundle.
2018-09-04 11:44:26.500136: I tensorflow/cc/saved_model/loader.cc:233] SavedModel load for tags { serve }; Status: success. Took 101670 microseconds.
Output:
{
  "classes": [
    [
      "0",
      "1"
    ]
  ],
  "scores": [
    [
      0.54612064,
      0.45387936
    ]
  ]
}
```