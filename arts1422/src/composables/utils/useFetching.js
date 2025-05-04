import axios from 'axios';

export function doFetch(address, callback, errorCatch, method = 'get') {
  return async (args) =>
    await axios[method](import.meta.env.VITE_API_BASE + address, args)
      .then(callback)
      .catch(errorCatch);
}

export function doGet(address, callback, errorCatch) {
  return doFetch(address, callback, errorCatch);
}

export function doPost(address, callback, errorCatch) {
  return doFetch(address, callback, errorCatch, 'post');
}
