import { ref } from 'vue';
import useSnackbar from '@/composables/global/useSnackbar';
import { doGet } from '@/composables/utils/useFetching';
import useChartState from '@/composables/charts/useChartState';

const { showSnackbar } = useSnackbar();
const { setHotspots } = useChartState();

const fetchHeatmapData = async () => {
  let data = [];
  await doGet(
    'heatmap',
    (response) => {
      data = response.data;
    },
    // eslint-disable-next-line no-unused-vars
    (_erorr) => {
      showSnackbar('Error fetching heatmap data', 'error');
    },
  )();
  return data;
};

const scaleFunction = (value) => {
  // return Math.exp(value / 1000);
  return value;
};

const heatmapData = ref([]);

const processHeatmap = async () => {
  const response = await fetchHeatmapData();
  let data = response.heatmap;
  // console.log(data);
  data = data[0].map((_, colIndex) => data.map((row) => row[colIndex]));
  data = data.map((row) => row.map((value) => scaleFunction(value)));
  // console.log(data);
  // console.log(data);
  // console.log(data.length);
  heatmapData.value = data;
  let initGeometry = response.init_geometry;
  setHotspots(initGeometry);
};

export default () => {
  processHeatmap();

  return {
    heatmapData,
  };
};
