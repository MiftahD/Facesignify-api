const { Storage } = require('@google-cloud/storage');
const tf = require('@tensorflow/tfjs-node');

const loadModel = async (bucketName, modelPath) => {
  const storage = new Storage();
  const [files] = await storage.bucket(bucketName).getFiles({ prefix: modelPath });

  const modelFile = files.find(file => file.name.endsWith('.json'));
  if (!modelFile) {
    throw new Error('Model file not found in bucket');
  }

  const modelUrl = `gs://${bucketName}/${modelFile.name}`;
  const model = await tf.loadGraphModel(modelUrl);

  return model;
};

module.exports = loadModel;
