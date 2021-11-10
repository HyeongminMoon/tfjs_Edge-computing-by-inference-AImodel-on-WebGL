// import * as tf from '@tensorflow/tfjs';
// import {loadGraphModel} from '@tensorflow/tfjs-converter';
// import * as tflite from '@tensorflow/tfjs-tflite';

tf.setBackend('webgl');
// tf.env().set("WEBGL_FORCE_F16_TEXTURES", true);
// tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', false);
// tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 0);
console.log(tf.getBackend());
console.log("hi");

// const MODEL_URL = 'aug4_10_180_rep16.tflite';
const MODEL_URL = 'aug4_10_180_tfjs_float16/model.json'
const cat = document.getElementById('cat');

console.time("preprocessing time");
const expanded_pixels = tf.tidy(() => {
    var pixels = tf.browser.fromPixels(cat);
    console.log(pixels.shape);
    pixels = tf.image.resizeBilinear(pixels, [180, 320]);//180, 320, 3
    pixels = pixels.div(pixels.max());
    
    var [r,g,b] = tf.split(pixels, 3, 2);
    r = r.sub(0.406).div(0.225);
    g = g.sub(0.456).div(0.224);
    b = b.sub(0.485).div(0.229);
    
    var normalized_pixels = tf.stack([r,g,b], 2);
    normalized_pixels = normalized_pixels.squeeze(3);
    normalized_pixels = tf.transpose(normalized_pixels, [2,0,1]);
    return normalized_pixels.expandDims(0);
});
console.timeEnd("preprocessing time");

console.log(expanded_pixels.shape);

model_predict = async (img) => {
    console.log("Start prediction");
    
    const model = await tf.loadGraphModel(MODEL_URL);
//     const model = await tflite.loadTFLiteModel(
//         MODEL_URL,
//         {numThreads: navigator.hardwareConcurrency / 2}
//     );
    console.log("loaded model")
    for (let i = 0; i < 10; i++) {
      
        
        
        const y = tf.tidy(() => {
            console.time('Prediction time');
            const predict = model.predict(img);
            console.timeEnd('Prediction time');
            console.time('postprocessing time');
//             var result = predict['PartitionedCall:0'];
//             var result = predict['Identity'];
            var result = predict[0];
            result = result.squeeze(1);
            const white = tf.ones(([1, 180, 320]));

            const pred = white.sub(result);
            const max = pred.max();
            const min = pred.min();
            const a = max.sub(min);
            const b = pred.sub(min);
            const dn = b.div(a)
            console.log(dn.shape)
            var tp_img = tf.transpose(dn, [1,2,0]);
            console.log(tp_img.shape);
        //     tp_img = tf.image.resizeBilinear(tp_img, [720, 1280]);
            console.timeEnd('postprocessing time');
            return tf.image.resizeBilinear(tp_img, [720, 1280]);
        });    
        const canvas = document.createElement('canvas');
        document.body.appendChild(canvas);
        canvas.height = 720;
        canvas.width = 1280;
        const final_img = tf.browser.toPixels(y, canvas);

        y.dispose();   
    }
//     return result
};

model_predict(expanded_pixels);


// n, h, w, c
const ones = tf.ones(([1, 3, 180, 320]))
const zeros = tf.zeros(([1, 3, 180, 320]))
// const IMAGENET_CLASSES = require('./imagenet_classes');

getTopKClasses = (logits, topK) => {
    const predictions = tf.tidy(() => {
        return tf.softmax(logits);
    });

    const values = predictions.dataSync();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList
                         .sort((a, b) => {
                           return b.value - a.value;
                         })
                         .slice(0, topK);

    return predictionList.map(x => {
      return {label: IMAGENET_CLASSES[x.index], value: x.value};
    });
}
