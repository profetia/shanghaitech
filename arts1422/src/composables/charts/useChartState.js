import { reactive, ref } from 'vue';

const initTimeRange = {
  clockRange: [0, 24],
  dateRange: [0, 186],
};

const timeRange = reactive({
  ...initTimeRange,
});

const hotspots = ref([]); // locations
const highlights = ref([]); // counties
const selected = reactive({
  id: 0,
});
const actualShow = ref([]);

const fetchStatus = ref(false);

const setTimeRange = (dateRange, clockRange) => {
  timeRange.clockRange = clockRange;
  timeRange.dateRange = dateRange;
};

const resetTimeRange = () => {
  timeRange.clockRange = initTimeRange.clockRange;
  timeRange.dateRange = initTimeRange.dateRange;
};

const setHotspots = (data) => {
  hotspots.value = data;
};

const setHighlights = (data) => {
  highlights.value = data;
};

const appendHighlights = (data) => {
  if (!highlights.value.includes(data)) {
    highlights.value.push(data);
  }
};

export default () => {
  return {
    timeRange,
    hotspots,
    highlights,
    fetchStatus,
    selected,
    actualShow,

    setTimeRange,
    resetTimeRange,
    setHotspots,
    setHighlights,
    appendHighlights,
  };
};
