# tfjs_Edge-computing-by-inference-AImodel-on-WebGL

Test tensorflow js for custom pretrained model(Pytorch U-2NET)

All valuable implements are in test.js


I recommend you to notice these references.
### reference 
* [Tensorflow.js](https://www.tensorflow.org/js)
* [Tfjs api](https://js.tensorflow.org/api/latest/)
* [pytorch to onnx](https://docs.microsoft.com/ko-kr/windows/ai/windows-ml/tutorials/pytorch-convert-model)
* [onnx to tensorflow](https://github.com/onnx/onnx-tensorflow)
* [tensorflow to tfjs](https://www.tensorflow.org/js/guide/conversion)
* [tensorflow lite quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

### What I did
* Convert pretrained Pytorch model to onnx
* Convert onnx model to tensorflow(saved model)
* Convert tensorflow model to tfjs, tflite
* Load tfjs model by Tensorflow.js Graphmodel
* Using WebGL backend, implement inference code
* Load tflite model by tflite-alpha module, but not implemented yet(see the issues below)
* Quantization Tfjs and Tflite

### TODO
I found [these](https://github.com/tensorflow/tfjs/issues/4166) [issues](https://github.com/tensorflow/tfjs/issues/5689), at now tfjs doesnt surpport gpu implementation for tensorflow lite.
So I need update so that I can use tflite models.(tensorflow js model can used)
