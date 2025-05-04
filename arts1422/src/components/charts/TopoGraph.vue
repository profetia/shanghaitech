<script setup>
import { d3RefNode } from '@/composables/d3/core/dreactive';
import D3Wrapper from '@/components/d3/core/D3Wrapper.vue';
import * as d3 from 'd3';
import { topoGraph } from '@/composables/d3/charts/topology';
import useTopoGraph from '@/composables/charts/useTopoGraph';
import useChartState from '@/composables/charts/useChartState';
import { toValidLength } from '@/composables/leaflet/utils/layout';
import { watch } from 'vue';
import { getHaikouByCode } from '@/composables/utils/useHaikou';

const props = defineProps({
  width: {
    type: [String, Number],
    default: 800,
  },
  height: {
    type: [String, Number],
    default: 588,
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
});

const { topoGraphData, actualShow } = useTopoGraph(props);
const { selected, highlights } = useChartState();

const chart = d3RefNode(() => {
  let scatter = topoGraph({
    ...props,
    data: (() => ({
      nodes: topoGraphData.value.nodes.filter((d) =>
        actualShow.value.includes(d.id),
      ),
      edges: topoGraphData.value.edges.filter((d) => {
        if (
          highlights.value.includes(String(d.x)) &&
          highlights.value.includes(String(d.y))
        ) {
          return (
            actualShow.value.includes(String(d.x)) &&
            actualShow.value.includes(String(d.y))
          );
        } else {
          return (
            actualShow.value.includes(String(d.x)) ||
            actualShow.value.includes(String(d.y))
          );
        }
      }),
    }))(),
    selectCallback: (_event, d) => {
      const adcode = d.id;
      selected.id = adcode;
    },
    // eslint-disable-next-line no-unused-vars
    unselectCallback: (_event, _d) => {
      selected.id = 0;
    },
  });

  return scatter.node();
});

watch(chart, (value) => {
  const scatter = d3.select(value);
  scatter.selectAll('circle').each((d, i, g) => {
    if (d.id == selected.id) {
      // console.log('selected', d);
      scatter
        .selectAll('.nodes')
        .attr('opacity', '0.1')
        .attr('isCalled', 'false');
      scatter
        .selectAll('.lines')
        .attr('opacity', '0.1')
        .attr('isCalled', 'false');
      scatter
        .selectAll(`#Topo_node_${d.id}`)
        .attr('opacity', '1')
        .attr('isCalled', 'true');
      scatter
        .selectAll(`.Topo_line_target_${d.id}`)
        .attr('opacity', '1')
        .attr('isCalled', 'false');
      scatter
        .selectAll(`.Topo_line_source_${d.id}`)
        .attr('opacity', '1')
        .attr('isCalled', 'false');
    }
  });
});

function handleRemoveHighlight(adcode) {
  highlights.value = highlights.value.filter((d) => d != adcode);
  actualShow.value = actualShow.value.filter((d) => d != adcode);
}
</script>
<template>
  <v-card variant="outlined">
    <div
      :style="{ width: toValidLength(width), height: toValidLength(height) }"
    >
      <D3Wrapper :node="chart" />
    </div>
    <div class="actualShow">
      <div v-for="(adcode, _index) in highlights" :key="adcode">
        <v-chip
          variant="outlined"
          label
          style="z-index: 10000"
          class="pl-1"
          :color="getHaikouByCode(adcode).properties.color"
          closable
          @click:close="handleRemoveHighlight(adcode)"
        >
          <v-checkbox
            hide-details
            v-model="actualShow"
            :label="getHaikouByCode(adcode).properties.name"
            :value="adcode"
            class="font-weight-bold"
          ></v-checkbox>
        </v-chip>
      </div>
    </div>
  </v-card>
</template>
<style scoped>
.actualShow {
  position: absolute;
  top: 4px;
  right: 4px;
  z-index: 10000;
}
</style>
