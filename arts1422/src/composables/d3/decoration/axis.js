import * as d3 from 'd3';

export function xAxis(
  x,
  {
    posX,
    posY,
    width,
    tickSpace = width / 80,
    label = '',
    labelX = width,
    lableY = posY + 15,
  } = {},
) {
  return (g) =>
    g
      .attr('transform', `translate(${posX}, ${posY})`)
      .call(d3.axisBottom(x).ticks(tickSpace))
      .call((g) => g.select('.domain').remove())
      .call((g) =>
        g
          .append('text')
          .attr('x', labelX)
          .attr('y', lableY)
          .attr('fill', 'currentColor')
          .attr('text-anchor', 'end')
          .text(label),
      );
}

export function yAxis(
  y,
  {
    posX,
    posY,
    height,
    tickSpace = height / 80,
    label = '',
    labelX = 10,
    lableY = 10,
  } = {},
) {
  return (g) =>
    g
      .attr('transform', `translate(${posX}, ${posY})`)
      .call(d3.axisLeft(y).ticks(tickSpace))
      .call((g) => g.select('.domain').remove())
      .call((g) =>
        g
          .append('text')
          .attr('x', labelX)
          .attr('y', lableY)
          .attr('fill', 'currentColor')
          .attr('text-anchor', 'start')
          .text(label),
      );
}
