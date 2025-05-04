<script setup>
import * as d3 from 'd3';
import { nextTick, ref, watch } from 'vue';
import { getUUID } from '@/composables/d3/utils/uuid';
import { toValidLength } from '@/composables/leaflet/utils/layout';

const props = defineProps({
  callback: {
    type: Function,
    required: true,
  },
  args: {
    // Any data that is reactive should be passed in here.
    // Otherwise, the callback will not be re-invoked.
    type: Object,
    default: {},
  },
  width: {
    type: [String, Number],
    default: 400,
  },
  height: {
    type: [String, Number],
    default: 300,
  },
});

const container = ref(null);
const uuid = getUUID();

function mountLeafletComponent() {
  let template = d3
    .create('div')
    .attr('class', 'leaflet-vue-component')
    .attr('id', `leaflet-vue-component-${uuid}`)
    .style('width', toValidLength(props.width))
    .style('height', toValidLength(props.height));

  container.value.appendChild(template.node());

  props.callback(template.node(), props.args);
}

nextTick(() => {
  mountLeafletComponent();
});

watch(
  () => props.args,
  () => {
    d3.select(`#leaflet-vue-component-${uuid}`).remove();
    mountLeafletComponent();
  },
);
</script>
<template>
  <div ref="container" style="width: 100%; height: 100%"></div>
</template>
