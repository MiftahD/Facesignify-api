const authController = require('../controllers/authController');

module.exports = [
  {
    method: 'POST',
    path: '/register',
    handler: authController.register,
    options: {
      auth: false, // Menonaktifkan autentikasi untuk route ini
    },
  },
  {
    method: 'POST',
    path: '/login/user',
    handler: authController.loginUser,
    options: {
      auth: false, // Menonaktifkan autentikasi untuk route ini
    },
  },
  {
    method: 'POST',
    path: '/login/admin',
    handler: authController.loginAdmin,
    options: {
      auth: false, // Menonaktifkan autentikasi untuk route ini
    },
  },
];
