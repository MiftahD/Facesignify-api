const errorMiddleware = (request, h) => {
  const response = request.response;
  if (response.isBoom) {
    const error = response;
    const statusCode = error.output.statusCode;
    const message = error.message;
    return h.response({ status: 'fail', message }).code(statusCode);
  }
  return h.continue;
};

module.exports = errorMiddleware;
