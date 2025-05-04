export function doDebounce(callback, delay = 200) {
  let timeoutID = undefined;
  return function () {
    const args = arguments;
    const that = this;
    clearTimeout(timeoutID);
    timeoutID = setTimeout(() => {
      callback.apply(that, args);
    }, delay);
  };
}

export function doSharedDebounce(callbacks, delays = 200) {
  let timeoutID = undefined;
  return (() => {
    let res = []
    for (let i = 0; i < callbacks.length; i++) {
      res.push(function () {  
        const that = this;
        const args = arguments;
        clearTimeout(timeoutID);
        timeoutID = setTimeout(() => {
          callbacks[i].apply(that, args);
        }, typeof delays === 'number' ? delays : delays[i]);
      })
    }
    return res
  })()
}