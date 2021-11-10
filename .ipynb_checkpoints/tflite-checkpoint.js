const tfliteModel = tflite.loadTFLiteModel(
     'https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/metadata/1');

const outputTensor = tf.tidy(() => {
    // Get pixels data from an image.
    const img = tf.browser.fromPixels(document.querySelector('img'));
    // Normalize (might also do resize here if necessary).
    const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
    // Run the inference.
    let outputTensor = tfliteModel.predict(input) as tf.Tensor;
    // De-normalize the result.
    return tf.mul(tf.add(outputTensor, 1), 127.5)
  });
console.log(outputTensor);
