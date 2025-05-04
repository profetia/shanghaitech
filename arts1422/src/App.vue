<script setup>
import Heatmap from '@/components/charts/Heatmap.vue';
import Geometry from '@/components/charts/Geometry.vue';
import TopoGraph from '@/components/charts/TopoGraph.vue';
import useSnackbar from '@/composables/global/useSnackbar';
import useChartState from '@/composables/charts/useChartState';

const { snackbar } = useSnackbar();

const { fetchStatus } = useChartState();
</script>
<template>
  <v-app>
    <v-main>
      <div class="d-flex justify-center pt-4">
        <div>
          <div class="d-flex justify-center">
            <div class="text-h2">HORRAS</div>
          </div>
          <div>
            <div>Haikou Online Ride-hailing Records Analysis System</div>
          </div>
        </div>
      </div>
      <v-divider></v-divider>
      <div class="py-2">
        <div class="d-flex justify-center">
          <div style="width: 1700px">
            <div
              class="d-flex justify-space-between"
              style="margin-left: 48px; margin-right: 20px"
            >
              <div>
                <Geometry />
              </div>
              <v-divider vertical></v-divider>
              <div>
                <TopoGraph />
              </div>
            </div>
          </div>
        </div>
      </div>
      <v-divider></v-divider>
      <div>
        <div class="d-flex justify-center">
          <div>
            <Heatmap />
          </div>
        </div>
      </div>
      <v-overlay
        :model-value="fetchStatus"
        class="align-center justify-center"
        persistent
      >
        <v-progress-circular
          indeterminate
          size="64"
          color="primary"
        ></v-progress-circular>
      </v-overlay>
      <v-snackbar
        v-model="snackbar.show"
        :color="snackbar.color"
        :timeout="snackbar.timeout"
      >
        {{ snackbar.message }}
      </v-snackbar>
    </v-main>
  </v-app>
</template>
