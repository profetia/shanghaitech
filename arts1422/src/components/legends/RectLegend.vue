<script setup>
import D3Wrapper from '@/components/d3/core/D3Wrapper.vue';
import { d3RefNode } from '@/composables/d3/core/dreactive';
import { rectLegend } from '@/composables/d3/legend/discrete';
import * as d3 from 'd3';

const props = defineProps({
  data: {
    type: Array,
    required: true,
  },
  rows: {
    type: [Number, String],
    default: 2,
  },
  columns: {
    type: [Number, String],
    default: null,
  },
  scheme: {
    type: Array,
    default: d3.schemeCategory10,
  },
  rectWidth: {
    type: [Number, String],
    default: 20,
  },
  rectHeight: {
    type: [Number, String],
    default: 20,
  },
  rectMargin: {
    type: [Number, String],
    default: 0,
  },
  textWidth: {
    type: [Number, String],
    default: 120,
  },
  width: {
    type: [Number, String],
    default: null,
  },
  height: {
    type: [Number, String],
    default: null,
  },
});

const legend = d3RefNode(() => {
  let {
    data,
    rows,
    columns,
    rectWidth,
    rectHeight,
    rectMargin,
    textWidth,
    width,
    height,
  } = props;

  if (columns === null) {
    columns = Math.floor(new Set(data.map((d) => d.category)).size / rows);
  }

  if (width === null) {
    width = columns * (rectWidth + 2 * rectMargin + textWidth);
  }

  if (height === null) {
    height = rows * (rectHeight + 2 * rectMargin);
  }

  let rects = rectLegend({
    ...props,
    columns,
    width,
    height,
  });

  return rects.node();
});
</script>
<template>
  <D3Wrapper :node="legend" />
</template>
