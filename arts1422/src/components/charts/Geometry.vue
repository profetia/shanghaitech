<script setup>
import useGeometry from '@/composables/charts/useGeometry';
import * as L from 'leaflet';
import LeafletWrapper from '@/components/leaflet/core/LeafletWrapper.vue';
import { heatmap } from '@/composables/leaflet/charts/heatmap';
import { polygon } from '@/composables/leaflet/charts/polygon';
import { getHaikouAll, getHaikouByCode } from '@/composables/utils/useHaikou';
import useChartState from '@/composables/charts/useChartState';
import { tileGaoDeNormal } from '@/composables/leaflet/tiles/provider';
import { watch } from 'vue';

const { geometry, chartConfig, mapState, lowResSample } = useGeometry();
const { appendHighlights, selected, actualShow } = useChartState();

const countyLayer = polygon({
  data: getHaikouAll(),
  // map: haikouMap,
  color: 'grey',
  eventHandlers: {
    mouseout: (e) => {
      const adcode = e.target.feature.properties.adcode;
      if (adcode == selected.id) {
        matchSelected(selected.id);
      } else if (actualShow.value.includes(adcode)) {
        matchSelected(adcode);
      } else {
        countyLayer.resetStyle(e.target);
      }
      matchSelected(selected.id);
      for (let adcode of actualShow.value) {
        matchSelected(adcode);
      }
    },
    click: (e) => {
      // haikouMap.fitBounds(e.target.getBounds());
      const adcode = e.target.feature.properties.adcode;
      matchSelected(adcode);
      appendHighlights(adcode);
    },
  },
});

function matchSelected(adcode) {
  for (let index in countyLayer._layers) {
    let layer = countyLayer._layers[index];
    // console.log(layer.feature.properties.adcode, value);
    if (layer.feature.properties.adcode == adcode) {
      layer.setStyle({
        weight: 5,
        color: getHaikouByCode(adcode).properties.color,
        dashArray: '',
        fillOpacity: 0.7,
      });
      layer.bringToFront();
    } else if (
      !actualShow.value.includes(layer.feature.properties.adcode) &&
      layer.feature.properties.adcode != selected.id
    ) {
      countyLayer.resetStyle(layer);
    }
  }
}

watch(selected, (value) => {
  matchSelected(value.id, value.color);
});

watch(actualShow, () => {
  for (let adcode of actualShow.value) {
    matchSelected(adcode);
  }
});

const initFn = (node, { geometry, chartConfig }) => {
  const baseLayer = L.tileLayer(...tileGaoDeNormal);

  // console.log(geometry);
  // console.log(geometry.data.filter((item) => item.count > geometry.max));

  const heatmapLayer = heatmap({
    data: geometry,
    useLocalExtrema: chartConfig.adoptive,
  });

  const haikouMap = L.map(node, {
    zoomControl: false,
    attributionControl: false,
    center: mapState.center,
    zoom: mapState.zoom,
    minZoom: 9,
    maxZoom: 18,
    maxBounds: [
      [19.2, 109.0],
      [20.8, 111.35],
    ],
    renderer: L.svg(),
    layers: [baseLayer, heatmapLayer],
  });

  haikouMap.on('zoomend', () => {
    mapState.zoom = haikouMap.getZoom();
  });

  haikouMap.on('moveend', () => {
    mapState.center = haikouMap.getCenter();
  });

  countyLayer.addTo(haikouMap);

  matchSelected(selected.id);
  for (let adcode of actualShow.value) {
    matchSelected(adcode);
  }
};
</script>
<template>
  <v-card variant="flat">
    <div>
      <LeafletWrapper
        :callback="initFn"
        :width="chartConfig.width"
        :height="chartConfig.height"
        :args="{ geometry, chartConfig }"
      />
    </div>
    <div style="margin-top: -100px" class="d-flex justify-space-between">
      <div class="d-flex flex-column justify-end">
        <v-chip
          label
          variant="elevated"
          color="teal"
          text-color="white"
          style="z-index: 10000"
          class="ml-3 mb-3 pl-1"
        >
          <v-checkbox
            hide-details
            v-model="chartConfig.relative"
            label="Relative Sample"
            class="font-weight-bold"
          ></v-checkbox>
        </v-chip>
        <v-chip
          label
          variant="elevated"
          color="purple"
          text-color="white"
          style="z-index: 10000"
          class="ml-3 mb-3 pl-1"
        >
          <v-checkbox
            hide-details
            v-model="lowResSample"
            label="Low-Res Sample"
            class="font-weight-bold"
          ></v-checkbox>
        </v-chip>
      </div>
      <div class="d-flex flex-column">
        <v-chip
          label
          variant="elevated"
          color="warning"
          style="z-index: 10000"
          class="mr-3 mb-3 pl-1"
        >
          <v-checkbox
            hide-details
            v-model="chartConfig.layers"
            label="Departures Layer"
            value="departures"
            class="font-weight-bold"
          ></v-checkbox>
        </v-chip>
        <v-chip
          label
          variant="elevated"
          color="info"
          style="z-index: 10000"
          class="mr-3 mb-3 pl-1"
        >
          <v-checkbox
            hide-details
            v-model="chartConfig.layers"
            label="Arrivals Layer"
            value="arrivals"
            class="font-weight-bold"
          ></v-checkbox>
        </v-chip>
      </div>
    </div>
  </v-card>
</template>
