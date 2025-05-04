import * as d3 from 'd3';

export function rectLegend({
  data,
  rows = 2,
  columns = Math.floor(new Set(data.map((d) => d.category)).size / rows),
  scheme = d3.schemeCategory10,
  rectWidth = 20,
  rectHeight = 20,
  rectMargin = 0,
  textWidth = 120,
  width = columns * (rectWidth + 2 * rectMargin + textWidth),
  height = rows * (rectHeight + 2 * rectMargin),
} = {}) {
  const svg = d3
    .create('svg')
    .attr('viewBox', [0, 0, width, height])
    .attr('width', width)
    .attr('height', height);
  const color = d3.scaleOrdinal(
    data.map((d) => d.category),
    scheme,
  );

  svg
    .selectAll('rect')
    .data(color.domain().slice())
    .join('rect')
    .attr(
      'x',
      (d, i) => (i % columns) * (rectWidth + 2 * rectMargin + textWidth),
    )
    .attr(
      'y',
      (d, i) => Math.floor(i / columns) * (rectHeight + 2 * rectMargin),
    )
    .attr('width', 20)
    .attr('height', 20)
    .attr('fill', (d) => color(d));

  svg
    .selectAll('text')
    .data(color.domain().slice())
    .join('text')
    .attr(
      'x',
      (d, i) => (i % columns) * (rectWidth + 2 * rectMargin + textWidth) + 25,
    )
    .attr(
      'y',
      (d, i) => Math.floor(i / columns) * (rectHeight + 2 * rectMargin) + 15,
    )
    .text((d) => d);

  return svg;
}
