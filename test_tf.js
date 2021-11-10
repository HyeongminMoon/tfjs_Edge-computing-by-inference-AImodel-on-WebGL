// import * as tf from '@tensorflow/tfjs';
// import {loadGraphModel} from '@tensorflow/tfjs-converter';
// import * as tflite from '@tensorflow/tfjs-tflite';


model_predict = async (img) => {
    console.log("Start prediction");
    
    const model = await tf.loadGraphModel(MODEL_URL);
    
    for (let i = 0; i < 1; i++) {
      
    console.time('Prediction time');
    var predict = model.predict(img);
//         console.log(typeof(predict));
//     result = predict[3].print()
        
//     console.log(predict['PartitionedCall:0'].print())
    result = predict[3];
//     console.log(typeof(predict[3]))
    result.print()
//     console.log(result.shape)
    
    console.timeEnd('Prediction time');
//     };
    
    const reversed_img = result.squeeze(1);
    const white = tf.ones(([1, 180, 320]));
    
    const pred = white.sub(reversed_img);
    const max = pred.max();
    const min = pred.min();
    const a = max.sub(min);
    const b = pred.sub(min);
    const dn = b.div(a)
    
    console.log(dn.shape)
//     const sq_img = result.squeeze(0);
//     console.log(sq_img.shape)
    
    var tp_img = tf.transpose(dn, [1,2,0]);
    console.log(tp_img.shape);
//     const int_img = tp_img.toInt();
    
    tp_img = tf.image.resizeBilinear(tp_img, [720, 1280]);
        
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    canvas.height = 720;
    canvas.width = 1280;
    const final_img = tf.browser.toPixels(tp_img, canvas);
    
//     model.dispose()
        
    }
    return result
};

const MODEL_URL = 'aug4_10_180_tfjs/model.json';

const cat = document.getElementById('cat');
const pixels = tf.browser.fromPixels(cat);

console.log(pixels.shape);

var resized_pixels = tf.image.resizeBilinear(pixels, [180, 320]);//180, 320, 3

resized_pixels = resized_pixels.div(resized_pixels.max());

// img[:,:,2] = (img[:,:,2]-0.406)/0.225
// img[:,:,1] = (img[:,:,1]-0.456)/0.224
// img[:,:,0] = (img[:,:,0]-0.485)/0.229
var [r,g,b] = tf.split(resized_pixels, 3, 2);


r = r.sub(0.406).div(0.225);
g = g.sub(0.456).div(0.224);
b = b.sub(0.485).div(0.229);

var normalized_pixels = tf.stack([r,g,b], 2);
normalized_pixels = normalized_pixels.squeeze(3);

const transposed_pixels = tf.transpose(normalized_pixels, [2,0,1]);
const expanded_pixels = transposed_pixels.expandDims(0);

console.log(expanded_pixels.shape);

result = model_predict(expanded_pixels);

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
