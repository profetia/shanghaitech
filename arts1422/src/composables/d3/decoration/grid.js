import * as d3 from 'd3';

export function naiveGrid(
  x,
  y,
  {
    posX1,
    posX2,
    posY1,
    posY2,
    stroke = 'currentColor',
    strokeOpacity = 0.1,
  } = {},
) {
  return (g) =>
    g
      .attr('stroke', stroke)
      .attr('stroke-opacity', strokeOpacity)
      .call((g) =>
        g
          .append('g')
          .selectAll('line')
          .data(x.ticks())
          .join('line')
          .attr('x1', (d) => 0.5 + x(d))
          .attr('x2', (d) => 0.5 + x(d))
          .attr('y1', posY1)
          .attr('y2', posY2),
      )
      .call((g) =>
        g
          .append('g')
          .selectAll('line')
          .data(y.ticks())
          .join('line')
          .attr('y1', (d) => 0.5 + y(d))
          .attr('y2', (d) => 0.5 + y(d))
          .attr('x1', posX1)
          .attr('x2', posX2),
      );
}
