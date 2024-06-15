const faceRecognitionService = require('../services/faceRecognitionService');
const InputError = require('../exceptions/InputError');

const predict = async (request, h) => {
  const { image } = request.payload;
  if (!image) {
    throw new InputError('Missing required file');
  }

  try {
    const result = await faceRecognitionService.predict(image._data);
    return h.response({ status: 'success', result }).code(200);
  } catch (error) {
    console.error('Prediction error:', error.message);
    return h.response({ status: 'fail', message: 'Failed to process image: ' + error.message }).code(500);
  }
};

module.exports = { predict };
