<script setup>
/*
@warning: This component is an inheritage of Clarivy's homework 2, it is not recommended because it is not hygienic, 
easy to understand, and has a bad performance. It is recommended to use the BrushScatter component instead.
*/
import * as d3 from 'd3';
import { d3RefNode } from '@/composables/d3/core/dreactive';
import D3Wrapper from '@/components/d3/core/D3Wrapper.vue';
import { d3ProxyGroupNodes } from '@/composables/d3/core/dproxy';
import { basicScatter } from '@/composables/d3/charts/scatter';
import useGroupedScatter from '@/composables/charts/useGroupedScatter';
import { computed, watch } from 'vue';

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

const { brushed, uid, doSetCursor } = useGroupedScatter(props);

let brush = d3.brush();

const selected = Array.from({ length: props.data.length }, () => false);

const chart = d3RefNode(() => {
  const { width, height, margin } = props;

  let scatter = basicScatter(props);

  let circles = scatter.selectAll('circle');
  d3ProxyGroupNodes(circles, (d, i, g) => `brush-${i}`);

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
    .on('brush', ({ selection }) => {
      if (selection) {
        const [[x0, y0], [x1, y1]] = selection;
        circles.each((d, i, g) => {
          const node = d3.select(g[i]);
          if (
            x0 < node.attr('cx') &&
            node.attr('cx') < x1 &&
            y0 < node.attr('cy') &&
            node.attr('cy') < y1
          ) {
            if (!selected[i]) {
              selected[i] = true;
              d3.selectAll(`.brush-${i}`).classed('selected', true);
              d3.selectAll(`.brush-${i}`).classed('unselected', false);
            }
          } else if (selected[i]) {
            selected[i] = false;
            d3.selectAll(`.brush-${i}`).classed('unselected', true);
            d3.selectAll(`.brush-${i}`).classed('selected', false);
          }
        });
      }
    })
    .on('end', ({ selection }) => {
      if (selection == null) {
        d3.selectAll('.selected')
          .classed('selected', false)
          .classed('unselected', true);
        selected.fill(false);
      }
    });

  scatter.call(brush);

  return scatter.node();
});

watch(
  () => brushed.cursor,
  (to, from) => {
    if (brushed.cursor && brushed.cursor != uid) {
      d3.select(chart.value).call(brush.move, null);
    }
  },
);
</script>
<template>
  <D3Wrapper :node="chart" />
</template>
<style>
.selected {
  fill: #8dd3c7;
  opacity: 1;
}

.unselected {
  fill: rgb(135, 135, 135);
  opacity: 0.7;
}
</style>
