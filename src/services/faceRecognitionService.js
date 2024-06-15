const tf = require('@tensorflow/tfjs-node');
const { createCanvas } = require('canvas');
const sharp = require('sharp');
const { IMAGENET_CLASSES } = require('../config/imagenet_classes');
const {db, admin} = require('../config/firebaseConfig');
require('dotenv').config();

async function loadModel() {
  try {
    const model = await tf.loadGraphModel(process.env.MODEL_URL);
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    throw new Error('Failed to load model');
  }
}

const predict = async (imageBuffer) => {
  const model = await loadModel();

  try{
  const processedImageBuffer = await sharp(imageBuffer)
    .resize(160, 160) // Resize to 160x160 pixels
    .toFormat('jpeg')
    .toBuffer();

  const imageTensor = tf.node.decodeImage(processedImageBuffer, 3)
    .expandDims(0)
    .toFloat()
    .div(tf.scalar(255));

    console.log('Image tensor shape:', imageTensor.shape);

  const predictions = await model.predict(imageTensor).data();

  const result = Array.from(predictions)
    .map((p, i) => ({
      probability: p,
      className: IMAGENET_CLASSES[i],
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 1);

    await savePrediction(result);

    return result;
}catch (error) {
  console.error('Error during prediction:', error);
  throw new Error('Failed to process image: ' + error.message);
}
};

const savePrediction= async (result) => {
  try {
    const predictionData = {
      result,
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
    };
    await db.collection('predictions').add(predictionData);
  } catch (error) {
    console.error('Error saving prediction to Firestore:', error);
    throw new Error('Failed to save prediction to Firestore: ' + error.message);
  }
};

module.exports = { predict };
