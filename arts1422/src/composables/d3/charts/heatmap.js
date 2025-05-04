/* eslint-disable no-unused-vars */
import * as d3 from 'd3';
import useChartState from '@/composables/charts/useChartState';
import { doDebounce } from '@/composables/utils/useDebounce';
import { daysOfMonths, monthNameShort } from '@/composables/utils/useDatetime';

const { setTimeRange, resetTimeRange } = useChartState();

export function defineAxis({
  xType = d3.scaleLinear(),
  yType = d3.scaleLinear(),
  xDomain,
  yDomain,
  margin,
  width,
  height,
}) {
  const x = xType.domain(xDomain).range([margin.left, width - margin.right]);
  const y = yType.domain(yDomain).range([margin.top, height - margin.bottom]);
  return { x, y };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function naiveHeatmap(
  {
    width = 400,
    height = 900,
    margin = {
      top: 10,
      right: 30,
      bottom: 10,
      left: 30,
    },
    heatmapColor = d3.interpolateRdPu,
    // d3.interpolateCubehelixLong('#f0f9e8', '#08306b')
    // d3.interpolateHslLong(d3.hsl(0, 0, 0.5), d3.hsl(0, 0, 0.9))
    // d3.interpolateLab('#ffffff', '#ff0000')
  },
  data,
) {
  if (data.length === 0) {
    return d3.create('svg');
  }
  const color = d3.scaleSequentialPow(
    [
      0,
      data.length
        ? d3.max(data, (d) => {
            return d3.max(d);
          })
        : 0,
    ],
    heatmapColor,
  );

  const { x, y } = defineAxis({
    xDomain: [0, data[0].length + 2],
    yDomain: [0, data.length + 2],
    ...arguments[0],
  });

  const svg = d3
    .create('svg')
    .attr('viewBox', [0, 0, width, height])
    .attr('width', width)
    .attr('height', height);

  svg
    .append('g')
    .attr('id', 'heatmap')
    .selectAll('g')
    .data(data)
    .join('g')
    .attr('cy', (d, i) => y(i + 1))
    .attr('index_y', (d, i) => i)
    .attr('transform', (d, i) => {
      return `translate(0,${y(i + 1)})`;
    })
    .selectAll('.naives')
    .data((d) => d)
    .join('rect')
    .attr('class', 'naives')
    .attr('isCell', true)
    .attr('x', (d, i) => {
      return x(i + 1);
    })
    .attr('index_x', (d, i) => i)
    .attr('cy', function (d, i, g) {
      return this.parentNode.getAttribute('cy');
    })
    .attr('index_y', function (d, i, g) {
      return this.parentNode.getAttribute('index_y');
    })
    .attr('width', (d, i) => x(2) - x(1) - 1)
    .attr('height', (d, i) => y(2) - y(1) - 1)
    .attr('fill', (d) => color(d));

  return svg;
}

export function brushedHeatmap(
  {
    width = 400,
    height = 900,
    margin = {
      top: 20,
      right: 30,
      bottom: 40,
      left: 30,
    },
    heatmapColor = d3.interpolateRdPu,
  },
  data,
) {
  let svg = naiveHeatmap(...arguments);

  // A dict to indicate whether a cell is selected
  let isSelected = {};

  if (data.length === 0) {
    return svg;
  }

  const { x, y } = defineAxis({
    xDomain: [0, data[0].length],
    yDomain: [0, data.length],
    ...arguments[0],
  });

  const updateSelect = (x0, x1, y0, y1) => {
    let xRange = [data[0].length, 0];
    let yRange = [data.length, 0];

    svg.selectAll('.naives').each((d, i, g) => {
      const node = d3.select(g[i]);
      node.attr('opacity', '0.5');
      if (!node.attr('isCell')) {
        return;
      }
      const currentX = node.attr('x');
      const currentY = node.attr('cy');
      const index_x = node.attr('index_x');
      const index_y = node.attr('index_y');

      if (
        currentX >= x0 &&
        currentX <= x1 &&
        currentY >= y0 &&
        currentY <= y1
      ) {
        node.attr('opacity', '1');
        isSelected[`${currentX},${currentY}`] = true;
        xRange[0] = Math.min(xRange[0], index_x);
        xRange[1] = Math.max(xRange[1], index_x);
        yRange[0] = Math.min(yRange[0], index_y);
        yRange[1] = Math.max(yRange[1], index_y);
        // node.attr('filter', 'brightness(50%)');
      } else {
        if (isSelected[`${currentX},${currentY}`]) {
          node.attr('opacity', '0.5');
          //node.attr('filter', 'brightness(100%)');
          isSelected[`${currentX},${currentY}`] = false;
        }
      }
    });
    return { xRange, yRange };
  };

  const brushstart = () => {
    svg.node().focus();
  };

  const brushmove = doDebounce((event) => {
    const selection = event.selection;
    if (!selection) {
      svg.selectAll('.naives').attr('opacity', '1');
      return;
    }
    const [[x0, y0], [x1, y1]] = selection;

    updateSelect(x0, x1, y0, y1);
  }, 50);

  const brushend = (event) => {
    const selection = event.selection;
    if (!selection) {
      svg.select('#day_up').text(`May 1-Oct 31`);
      svg.select('#hour').text(`0:00-24:00`);
      svg.selectAll('.naives').attr('opacity', '1');
      resetTimeRange();
      return;
    }

    const [[x0, y0], [x1, y1]] = selection;

    const { xRange, yRange } = updateSelect(x0, x1, y0, y1);

    let xRange_time = [];
    xRange.forEach((d) => {
      let m = 0;
      let days = d + 1;
      while (days > daysOfMonths[m]) {
        days -= daysOfMonths[m];
        m += 1;
      }
      xRange_time.push({
        month: m + 5,
        day: days,
      });
    });
    svg
      .select('#day_up')
      .text(
        `${monthNameShort[xRange_time[0].month - 1]} ${xRange_time[0].day}-${
          monthNameShort[xRange_time[1].month - 1]
        } ${xRange_time[1].day}`,
      );

    svg.select('#hour').text(`${yRange[0]}:00-${yRange[1] + 1}:00`);
    setTimeRange(xRange, yRange);
  };
  // console.log(margin);
  let brush = d3
    .brush()
    .extent([
      [margin.top, 0],
      [width - margin.right, height - margin.bottom],
    ])
    .on('start', brushstart)
    .on('brush', brushmove)
    .on('end', brushend);

  svg.select('g').call(brush);

  return svg;
}
