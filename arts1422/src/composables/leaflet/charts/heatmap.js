import HeatmapOverlay from 'heatmap.js/plugins/leaflet-heatmap/leaflet-heatmap.js';

export function heatmap({
  data,
  radius = 0.008,
  opacity = 0.85,
  maxOpacity = 0.8,
  minOpacity = 0.1,
  blur = 0.85,
  latField = 'lat',
  lngField = 'lng',
  valueField = 'count',
  useLocalExtrema = false,
} = {}) {
  const config = {
    radius,
    blur,
    latField,
    lngField,
    valueField,
    scaleRadius: true,
    useLocalExtrema,
  };

  if (maxOpacity && minOpacity) {
    config.maxOpacity = maxOpacity;
    config.minOpacity = minOpacity;
  } else {
    config.opacity = opacity;
  }

  const heatmapLayer = new HeatmapOverlay(config);
  heatmapLayer.setData(data);

  return heatmapLayer;
}
