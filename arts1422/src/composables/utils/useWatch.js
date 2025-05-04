import { watch } from 'vue';

export const recordWatch = (src, callback, options = {}) => {
  let oldValue = Object.assign({}, src);
  watch(
    src,
    function () {
      callback.apply(this, [oldValue, ...arguments]);
      oldValue = Object.assign({}, src);
    },
    options,
  );
};
