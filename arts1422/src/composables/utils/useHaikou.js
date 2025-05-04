import GeoJson460105 from '@/assets/geojson/460105.json';
import GeoJson460106 from '@/assets/geojson/460106.json';
import GeoJson460107 from '@/assets/geojson/460107.json';
import GeoJson460108 from '@/assets/geojson/460108.json';
import * as d3 from 'd3';

const registry = {
  460105: GeoJson460105,
  460106: GeoJson460106,
  460107: GeoJson460107,
  460108: GeoJson460108,
};

function formatToTwoDigits(num) {
  return num < 10 ? `0${num}` : num;
}

(function init() {
  for (const key in registry) {
    registry[key].features.forEach((feature, index) => {
      feature.properties = {
        ...feature.properties,
        adcode: `${key}${formatToTwoDigits(index)}`,
      };
      feature.properties.color =
        d3.schemeCategory10[feature.properties.adcode % 10];
    });
  }
})();

export function getHaikouAll() {
  const haikouAll = {
    type: 'FeatureCollection',
    features: [],
  };
  for (const key in registry) {
    haikouAll.features.push(...registry[key].features);
  }
  return haikouAll;
}

export function getHaikouByRegion(region) {
  return registry[region];
}

export function getHaikouByCode(code) {
  code = new String(code);
  const regionID = code.substring(0, 6);
  const polygonID = code.substring(6);
  return registry[regionID].features[new Number(polygonID)];
}
