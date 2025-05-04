import { computed } from 'vue';

export function d3RefNode(callback) {
  return computed(callback);
}
