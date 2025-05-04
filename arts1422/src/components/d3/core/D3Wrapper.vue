<script setup>
import * as d3 from 'd3';
import { nextTick, ref, watch } from 'vue';
import { getUUID } from '@/composables/d3/utils/uuid';

const props = defineProps({
  node: {
    type: Node,
    required: true,
  },
});

const container = ref(null);
const uuid = getUUID();

function mountD3Component() {
  let template = d3
    .create('div')
    .attr('class', 'd3-component')
    .attr('id', `d3-component-${uuid}`);

  template.append(() => props.node);

  container.value.appendChild(template.node());
}

nextTick(() => {
  mountD3Component();
});

watch(
  () => props.node,
  () => {
    d3.select(`#d3-component-${uuid}`).remove();
    mountD3Component();
  },
);
</script>
<template>
  <div ref="container"></div>
</template>
