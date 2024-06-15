const faceRecognitionController = require('../controllers/faceRecognitionController');

module.exports = [
  {
    method: 'POST',
    path: '/predict',
    handler: faceRecognitionController.predict,
    options: {
      payload: {
        output: 'stream',
        parse: true,
        allow: 'multipart/form-data',
        multipart: true,
        maxBytes: 10485760, // 10 MB
      },
    },
  },
];
