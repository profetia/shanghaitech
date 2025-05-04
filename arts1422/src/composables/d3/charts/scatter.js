import * as d3 from 'd3';
import { xAxis, yAxis } from '@/composables/d3/decoration/axis';
import { naiveGrid } from '@/composables/d3/decoration/grid';

export function naiveScatter({
  data,
  width = 800,
  height = 600,
  margin = {
    top: 20,
    right: 30,
    bottom: 40,
    left: 30,
  },
  xType = d3.scaleLinear,
  yType = d3.scaleLinear,
  xDomain = [0, 1],
  yDomain = [0, 1],
  scheme = d3.schemeCategory10,
}) {
  const svg = d3
    .create('svg')
    .attr('viewBox', [0, 0, width, height])
    .attr('width', width)
    .attr('height', height);

  const x = xType.domain(xDomain).range([margin.left, width - margin.right]);
  const y = yType.domain(yDomain).range([height - margin.bottom, margin.top]);

  const color = d3.scaleOrdinal(
    data.map((d) => d.category),
    scheme,
  );

  svg
    .append('g')
    .attr('stroke-width', 1.5)
    .attr('font-family', 'sans-serif')
    .attr('font-size', 10)
    .selectAll('circle')
    .data(data)
    .join('circle')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('fill', (d) => color(d.category))
    .attr('opacity', 0.7)
    .attr('z-index', 1)
    .attr('r', 3);

  return svg;
}

export function basicScatter({
  data,
  width = 800,
  height = 600,
  margin = {
    top: 20,
    right: 30,
    left: 30,
    bottom: 40,
  },
  xType = d3.scaleLinear,
  yType = d3.scaleLinear,
  xDomain = [0, 1],
  yDomain = [0, 1],
  xLabel = '',
  yLabel = '',
  scheme = d3.schemeCategory10,
} = {}) {
  let scatter = naiveScatter({
    data,
    width,
    height,
    margin,
    xType,
    yType,
    xDomain,
    yDomain,
    scheme,
  });

  const x = xType.domain(xDomain).range([margin.left, width - margin.right]);
  const y = yType.domain(yDomain).range([height - margin.bottom, margin.top]);

  scatter.append('g').call(
    xAxis(x, {
      posX: 0,
      posY: height - margin.bottom,
      width: width,
      label: xLabel,
      labelX: width,
      labelY: margin.bottom - 4,
    }),
  );

  scatter.append('g').call(
    yAxis(y, {
      posX: margin.left,
      posY: 0,
      height: height,
      label: yLabel,
      labelX: -margin.left,
      labelY: 10,
    }),
  );

  scatter.append('g').call(
    naiveGrid(x, y, {
      posX1: margin.left,
      posY1: margin.top,
      posX2: width - margin.right,
      posY2: height - margin.bottom,
    }),
  );

  return scatter;
}
