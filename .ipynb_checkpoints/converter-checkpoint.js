const cat = document.getElementById('cat');
const pixels = tf.browser.fromPixels(cat);

console.log(pixels.shape);
// console.log(pixels.print());

var resized_pixels = tf.image.resizeBilinear(pixels, [360, 640]);

console.log(resized_pixels.shape);

// img[:,:,2] = (img[:,:,2]-0.406)/0.225
// img[:,:,1] = (img[:,:,1]-0.456)/0.224
// img[:,:,0] = (img[:,:,0]-0.485)/0.229

resized_pixels = resized_pixels.div(resized_pixels.max());

var [r,g,b] = tf.split(resized_pixels, 3, 2);


r = r.sub(0.406).div(0.225);
g = g.sub(0.456).div(0.224);
b = b.sub(0.485).div(0.229);
// b = b.div(0.229)

var normalized_pixels = tf.stack([r,g,b], 2);
normalized_pixels = normalized_pixels.squeeze(3) 

console.log(normalized_pixels.shape)


// var x = tf.tensor3d([1,2,3,4,5,6,7,8,9,10,11,12],[2,2,3])
// x.print();
// var [r,g,b] = tf.split(x,3,2)
// x = tf.stack([r,g,b], 2)
// console.log(x.shape)
// x = x.squeeze(3);
// x.print();