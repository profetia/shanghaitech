<!-- eslint-disable vue/multi-word-component-names -->
<script setup>
import { d3RefNode } from '@/composables/d3/core/dreactive';
import D3Wrapper from '@/components/d3/core/D3Wrapper.vue';
import * as d3 from 'd3';
import { brushedHeatmap } from '@/composables/d3/charts/heatmap';
import {
  weatherExtension,
  differentialExtension,
  axisExtension,
  datetimeExtension,
} from '@/composables/d3/charts/extension';
import useHeatmap from '@/composables/charts/useHeatmap';
import useChartState from '@/composables/charts/useChartState';

const props = defineProps({
  width: {
    type: [String, Number],
    default: 1700,
  },
  height: {
    type: [String, Number],
    default: 350,
  },
  margin: {
    type: Object,
    default: () => ({
      top: 35,
      right: 100,
      left: 70,
      bottom: 140,
    }),
  },
});

const { heatmapData } = useHeatmap(props);
const { timeRange } = useChartState();

const rawChart = d3RefNode(() => {
  let scatter = brushedHeatmap(props, heatmapData.value);
  scatter = axisExtension({ svg: scatter, ...props }, heatmapData.value);
  scatter = datetimeExtension({ svg: scatter, ...props });
  scatter = weatherExtension({ svg: scatter, ...props }, heatmapData.value);
  return scatter.node();
});

const chart = d3RefNode(() => {
  let scatter = d3.select(rawChart.value);
  scatter = differentialExtension(
    { svg: scatter, ...props },
    heatmapData.value.map((row, i) => {
      return row.map((unit, j) => {
        if (
          timeRange.clockRange[0] <= i &&
          i <= timeRange.clockRange[1] &&
          timeRange.dateRange[0] <= j &&
          j <= timeRange.dateRange[1]
        ) {
          return unit;
        } else {
          return 0;
        }
      });
    }),
  );
  scatter.style('font-size', '12px');
  return scatter.node();
});
</script>
<template>
  <D3Wrapper :node="chart" />
</template>
