import * as L from 'leaflet';

export function polygon({
  data,
  color = 'white',
  opacity = 0.8,
  weight = 2,
  fillOpacity = 0.2,
  dashArray = '3',
  eventHandlers = {},
}) {
  const polygonLayer = L.geoJSON(data, {
    style: (feature) => ({
      fillColor:
        color instanceof Function ? color(feature.properties.value) : color,
      color: 'white',
      weight,
      opacity,
      fillOpacity,
      dashArray,
    }),
    onEachFeature: (feature, layer) => {
      // console.log(feature, layer);
      layer.on({
        mouseover: (e) => {
          const layer = e.target;
          layer.setStyle({
            weight: 5,
            color: '#666',
            dashArray: '',
            fillOpacity: 0.7,
          });
          layer.bringToFront();
        },
        mouseout: (e) => {
          polygonLayer.resetStyle(e.target);
        },
        ...eventHandlers,
      });
    },
  });
  // console.log(polygonLayer);
  return polygonLayer;
}
