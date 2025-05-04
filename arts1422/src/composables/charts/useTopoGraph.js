import { ref, watch } from 'vue';
import { doPost } from '@/composables/utils/useFetching';
import useChartState from '@/composables/charts/useChartState';
import useSnackbar from '@/composables/global/useSnackbar';
import { doDebounce } from '@/composables/utils/useDebounce';
import { recordWatch } from '@/composables/utils/useWatch';

const { highlights, timeRange, fetchStatus, actualShow } = useChartState();
const { showSnackbar } = useSnackbar();

const topoGraphData = ref({
  nodes: [],
  edges: [],
});

const fetchTopoGraphData = async () => {
  const dateLowerBound = 119;

  let data = {};
  await doPost(
    'topology',
    (response) => {
      data = response.data;
      fetchStatus.value = false;
    },
    // eslint-disable-next-line no-unused-vars
    (_erorr) => {
      showSnackbar('Error fetching hotspots data', 'error');
      fetchStatus.value = false;
      // console.log(_erorr);
    },
  )({
    nodes: highlights.value,
    date_range: [timeRange.dateRange[0] + dateLowerBound, timeRange.dateRange[1] + 1 + dateLowerBound],    
    clock_range: [
      timeRange.clockRange[0] * 60,
      (timeRange.clockRange[1] + 1) * 60,
    ],
  });
  return data;
};

const processTopoGraphData = async () => {
  fetchStatus.value = true;
  let data = await fetchTopoGraphData();
  topoGraphData.value = data;
};

recordWatch(
  highlights,
  doDebounce((oldValue, value) => {
    // console.log(value, oldValue._value);
    if (value.length < oldValue._value.length) {
      return;
    }
    processTopoGraphData();
    for (let county of highlights.value) {
      if (!actualShow.value.includes(county)) {
        actualShow.value.push(county);
      }
    }
  }),
  {
    deep: true,
  },
);

// eslint-disable-next-line no-unused-vars
watch(timeRange, (_value) => {
  if (highlights.value.length > 0) {
    processTopoGraphData();
  }
});

export default () => {
  // processTopoGraphData();

  return {
    topoGraphData,
    actualShow,
  };
};
