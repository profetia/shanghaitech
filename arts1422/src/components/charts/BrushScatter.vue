<script setup>
import * as d3 from 'd3';
import { d3RefNode } from '@/composables/d3/core/dreactive';
import D3Wrapper from '@/components/d3/core/D3Wrapper.vue';
import { d3ProxyBindNodes, d3ProxyArray } from '@/composables/d3/core/dproxy';
import { basicScatter } from '@/composables/d3/charts/scatter';
import useBrushScatter from '@/composables/charts/useBrushScatter';
import { computed, watch } from 'vue';
import { doDebounce } from '@/composables/utils/useDebounce';
import { difference } from '@/composables/utils/useSet';

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
    default: () => ({
      top: 20,
      right: 30,
      left: 30,
      bottom: 40,
    }),
  },
  xType: {
    default: d3.scaleLinear,
  },
  yType: {
    default: d3.scaleLinear,
  },
  xDomain: {
    type: Array,
    default: () => [0, 1],
  },
  yDomain: {
    type: Array,
    default: () => [0, 1],
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

const { brushed, uid, doUpdateBrush, doResetBrush, doSetCursor } =
  useBrushScatter(props);

const proxiedData = computed(() => d3ProxyArray(props.data));

let brush = d3.brush();

const chart = d3RefNode(() => {
  const { width, height, margin } = props;

  let scatter = basicScatter({
    ...props,
    data: proxiedData.value,
  });
  d3ProxyBindNodes(scatter.selectAll('circle'));

  brush = brush
    .extent([
      [margin.left, margin.top],
      [width - margin.right, height - margin.bottom],
    ])
    .on('start', ({ selection }) => {
      if (selection) {
        doSetCursor(uid);
      }
    })
    .on(
      'brush',
      doDebounce(({ selection }) => {
        if (selection) {
          const [[x0, y0], [x1, y1]] = selection;
          const selected = new Set();
          scatter.selectAll('circle').each((d, i, g) => {
            const node = d3.select(g[i]);
            if (
              x0 < node.attr('cx') &&
              node.attr('cx') < x1 &&
              y0 < node.attr('cy') &&
              node.attr('cy') < y1
            ) {
              selected.add(i);
            }
          });
          doUpdateBrush(selected);
        }
      }, 10),
    )
    .on('end', ({ selection }) => {
      if (selection == null) {
        doResetBrush();
      }
    });

  scatter.call(brush);

  return scatter.node();
});

watch(
  () => brushed.value,
  (to, from) => {
    if (to) {
      const selected = difference(to, from);
      const unselected = difference(from, to);

      for (const i of selected) {
        const node = d3.select(proxiedData.value[i].getD3Node());
        node.attr('fill', '#8dd3c7').attr('opacity', '1');
      }
      for (const i of unselected) {
        const node = d3.select(proxiedData.value[i].getD3Node());
        node.attr('fill', 'rgb(135, 135, 135)').attr('opacity', '0.7');
      }
    }
  },
);

watch(
  () => brushed.cursor,
  () => {
    if (brushed.cursor && brushed.cursor != uid) {
      d3.select(chart.value).call(brush.move, null);
    }
  },
);
</script>
<template>
  <D3Wrapper :node="chart" />
</template>
