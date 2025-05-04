import { computed, reactive, ref, watch } from 'vue';
import useChartState from '@/composables/charts/useChartState';
import { doPost } from '@/composables/utils/useFetching';
import useSnackbar from '@/composables/global/useSnackbar';

const { hotspots, timeRange, setHotspots, fetchStatus } = useChartState();
const { showSnackbar } = useSnackbar();

const chartConfig = reactive({
  relative: false,
  width: 800,
  height: 600,
  layers: ['departures', 'arrivals'],
});

const lowResSample = ref(true);

const mapState = reactive({
  center: [20.004658, 110.355043],
  zoom: 12,
});

const geometry = computed(() => {
  if (
    hotspots.value &&
    hotspots.value.length > 0 &&
    (chartConfig.layers.includes('departures') ||
      chartConfig.layers.includes('arrivals'))
  ) {
    const data = hotspots.value.map((hotspot) => ({
      ...hotspot,
      count: (() => {
        let res = 0;
        if (chartConfig.layers.includes('departures')) {
          res += hotspot.start_num;
        }
        if (chartConfig.layers.includes('arrivals')) {
          res += hotspot.dest_num;
        }
        if (!chartConfig.relative) {
          res = Math.log(res);
        }
        return res;
      })(),
    }));
    return {
      data,
      min: Math.max(
        0,
        (() => {
          let res = 0x7fffffff;
          for (let i = 0; i < data.length; i += 1) {
            res = Math.min(res, data[i].count);
          }
          return res;
        })(),
      ),
      max: Math.max(
        0,
        (() => {
          let res = 0;
          for (let i = 0; i < data.length; i += 1) {
            res = Math.max(res, data[i].count);
          }
          return res;
        })(),
      ),
    };
  } else {
    return {
      data: [],
      min: 0,
      max: 0,
    };
  }
});

const fetchHotspots = async () => {
  const dateLowerBound = 119;

  let data = [];
  await doPost(
    'geometry',
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
    sample: lowResSample.value,
    date_range: [timeRange.dateRange[0] + dateLowerBound, timeRange.dateRange[1] + 1 + dateLowerBound],    
    clock_range: [
      timeRange.clockRange[0] * 60,
      (timeRange.clockRange[1] + 1) * 60,
    ],
  });
  return data;
};

watch(
  timeRange,
  // eslint-disable-next-line no-unused-vars
  async (_value, _oldValue) => {
    fetchStatus.value = true;
    const hotspots = await fetchHotspots();
    setHotspots(hotspots);
  },
);

// eslint-disable-next-line no-unused-vars
watch(lowResSample, async (value) => {
  fetchStatus.value = true;
  const hotspots = await fetchHotspots();
  if (!value) {
    chartConfig.relative = true;
  } else {
    chartConfig.relative = false;
  }
  setHotspots(hotspots);
});

export default () => {
  return {
    lowResSample,
    geometry,
    chartConfig,
    mapState,
  };
};
