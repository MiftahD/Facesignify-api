const jwt = require('jsonwebtoken');
const { AuthenticationError } = require('../exceptions/AuthenticationError');
const { AuthorizationError } = require('../exceptions/AuthorizationError');


const authMiddleware = async (request, h) => {
  const authHeader = request.headers.authorization;
  if (!authHeader) {
    throw new AuthorizationError('Authorization header missing');
  }
  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    request.auth = { user: decoded };
    return h.continue;
  } catch (error) {
    throw new AuthenticationError('Invalid token');
  }
};

module.exports = authMiddleware;
