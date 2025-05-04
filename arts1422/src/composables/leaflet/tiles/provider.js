export const tileGaoDeNormal = [
  'http://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
  { subdomains: '1234' },
];

export const tileGaoDeSatellite = [
  'http://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
  { subdomains: '1234' },
];

export const tileTianDiNormal = [
  'http://t{s}.tianditu.gov.cn/vec_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=vec&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILECOL={x}&TILEROW={y}&TILEMATRIX={z}&tk=56b81006f361f6406d0e940d2f89a39c',
  { subdomains: ['0', '1', '2', '3', '4', '5', '6', '7'] },
];

export const tileGeoQNormal = [
  'http://map.geoq.cn/ArcGIS/rest/services/ChinaOnlineCommunity/MapServer/tile/{z}/{y}/{x}',
];

export const tileGeoQGrey = [
  'http://map.geoq.cn/ArcGIS/rest/services/ChinaOnlineStreetGray/MapServer/tile/{z}/{y}/{x}',
];

export const tileTecentRoad = [
  'http://rt0.map.gtimg.com/tile?z={z}&x={x}&y={-y}&type=vector&styleid=3&version=628',
];

export const tileOpenStreetNormal = [
  'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
];

export const tileArcgisSatellite = [
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
];

export const tileJwagRay = [
  'https://{s}.tile.jawg.io/jawg-matrix/{z}/{x}/{y}{r}.png?access-token=MOEv0c7HeIGtkp6COaRzlt1QhSeaYLo1QzHCHu8RUY4oMeY72DfQoyw9KxQmIIVL',
];

export const tileJwagNormal = [
  'https://{s}.tile.jawg.io/jawg-sunny/{z}/{x}/{y}{r}.png?access-token=MOEv0c7HeIGtkp6COaRzlt1QhSeaYLo1QzHCHu8RUY4oMeY72DfQoyw9KxQmIIVL',
];

export const tileThunderForestDark = [
  'https://{s}.tile.thunderforest.com/transport-dark/{z}/{x}/{y}.png?apikey=63bc80aa57164b0c94d24cba52c45bb4',
];
