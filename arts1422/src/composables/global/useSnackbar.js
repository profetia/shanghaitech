import { reactive } from 'vue';

const snackbar = reactive({
  show: false,
  message: '',
  timeout: 3000,
  color: 'success',
});

export default () => {
  const showSnackbar = (message, color = 'error', timeout = 3000) => {
    snackbar.message = message;
    snackbar.color = color;
    snackbar.timeout = timeout;
    snackbar.show = true;
  };

  return {
    snackbar,
    showSnackbar,
  };
};
