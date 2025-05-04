<script setup>
import * as d3 from 'd3';
import { d3RefNode } from '@/composables/d3/core/dreactive';
import { d3ProxyBindNodes, d3ProxyArray } from '@/composables/d3/core/dproxy';
import D3Wrapper from '@/components/d3/core/D3Wrapper.vue';
import { basicScatter } from '@/composables/d3/charts/scatter';
import { naiveTooltip } from '@/composables/d3/widget/tooltip';
import { computed } from 'vue';

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

const proxiedData = computed(() => d3ProxyArray(props.data));

const chart = d3RefNode(() => {
  let container = d3.create('div');
  let tooltip = naiveTooltip({
    width: 200,
    height: 325,
  });
  let scatter = basicScatter({
    ...props,
    data: proxiedData.value,
  });
  d3ProxyBindNodes(scatter.selectAll('circle'));

  scatter
    .selectAll('circle')
    .on('mouseover', (event, d) => {
      tooltip
        .html(() => {
          let res = ``;
          for (let key of Object.keys(d.unwrap()).filter(
            (key) =>
              key != 'x' &&
              key != 'y' &&
              key != 'category' &&
              key.indexOf('2d') == -1,
          )) {
            res = res + `<div>${key}: ${d[key]}</div>`;
          }
          return res;
        })
        .style('opacity', 1)
        .style('left', `${event.pageX + 20}px`)
        .style('top', `${event.pageY + 20}px`);
    })
    .on('mouseout', (event, d) => {
      tooltip.style('opacity', 0);
    });

  container.append(() => tooltip.node());
  container.append(() => scatter.node());

  return container.node();
});
</script>
<template>
  <D3Wrapper :node="chart" />
</template>
