import * as d3 from 'd3';

export function naiveTooltip({
  width = 120,
  height = 80,
  fontSize = 8,
  backgroundColor = 'white',
  borderStyle = 'solid',
  borderColor = 'black',
  borderWidth = 0.25,
}) {
  let tooltip = d3
    .create('div')
    .style('position', 'absolute')
    .style('width', `${width}px`)
    .style('height', `${height}px`)
    .style('font-size', `${fontSize}px`)
    .style('background-color', backgroundColor)
    .style('border-style', borderStyle)
    .style('border-color', borderColor)
    .style('border-width', `${borderWidth}px`)
    .style('opacity', 0);

  return tooltip;
}
