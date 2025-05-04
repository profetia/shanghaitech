<script setup>
import * as d3 from 'd3';
import { d3RefNode } from '@/composables/d3/core/dreactive';
import D3Wrapper from '@/components/d3/core/D3Wrapper.vue';

import { basicScatter } from '@/composables/d3/charts/scatter';

const props = defineProps({
  data: {
    type: Array,
    required: true,
  },
  width: {
    type: [String, Number],
    default: 800,
  },
  height: {
    type: [String, Number],
    default: 600,
  },
  margin: {
    type: Object,
    default: {
      top: 20,
      right: 30,
      left: 30,
      bottom: 40,
    },
  },
  xType: {
    default: d3.scaleLinear,
  },
  yType: {
    default: d3.scaleLinear,
  },
  xDomain: {
    type: Array,
    default: [0, 1],
  },
  yDomain: {
    type: Array,
    default: [0, 1],
  },
  xLabel: {
    type: String,
    default: '',
  },
  yLabel: {
    type: String,
    default: '',
  },
  scheme: {
    type: Array,
    default: d3.schemeCategory10,
  },
});

const chart = d3RefNode(() => {
  let scatter = basicScatter(props);

  return scatter.node();
});
</script>
<template>
  <D3Wrapper :node="chart" />
</template>
