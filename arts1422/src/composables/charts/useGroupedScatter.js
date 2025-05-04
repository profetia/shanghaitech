import { reactive, ref } from 'vue';

const brushed = reactive({
  cursor: null,
});
let uuid = 0;

export default () => {
  const doSetCursor = (who) => {
    brushed.cursor = who;
  };

  uuid += 1;
  const uid = uuid;

  return {
    brushed,
    uid,
    doSetCursor,
  };
};
