const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { Firestore } = require('@google-cloud/firestore');
const firestore = new Firestore();
const AuthenticationError = require('../exceptions/AuthenticationError');
const InputError = require('../exceptions/InputError');

const register = async (username, email, password, role) => {
  let userRef;

  if (role === 'student') {
    userRef = firestore.collection('users').doc(username);
  } else if (role === 'teacher') {
    userRef = firestore.collection('admins').doc(username);
  } else {
    throw new InputError('Invalid role');
  }

  const userDoc = await userRef.get();

  if (userDoc.exists) {
    throw new InputError('Username already exists');
  }

  const hashedPassword = await bcrypt.hash(password, 10);
  await userRef.set({ username, email, password: hashedPassword, role, createdAt: new Date(), updatedAt: new Date() });
};

const loginUser = async (username, password) => {
  const userRef = firestore.collection('users').doc(username);
  const userDoc = await userRef.get();
  if (!userDoc.exists) {
    throw new AuthenticationError('Invalid username or password');
  }
  const user = userDoc.data();
  const isValidPassword = await bcrypt.compare(password, user.password);
  if (!isValidPassword) {
    throw new AuthenticationError('Invalid username or password');
  }
  const token = jwt.sign({ username, role: user.role }, process.env.JWT_SECRET, { expiresIn: '4h' });
  return token;
};

const loginAdmin = async (username, password) => {
  const adminRef = firestore.collection('admins').doc(username);
  const adminDoc = await adminRef.get();
  if (!adminDoc.exists) {
    throw new AuthenticationError('Invalid username or password');
  }
  const admin = adminDoc.data();
  const isValidPassword = await bcrypt.compare(password, admin.password);
  if (!isValidPassword) {
    throw new AuthenticationError('Invalid username or password');
  }
  const token = jwt.sign({ username, role: admin.role }, process.env.JWT_SECRET, { expiresIn: '1h' });
  return token;
};

module.exports = { register, loginUser, loginAdmin };
