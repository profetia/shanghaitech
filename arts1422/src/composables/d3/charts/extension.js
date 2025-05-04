import { weatherHaikou, weatherTypes } from '@/composables/utils/useWeather';
import * as d3 from 'd3';
import { defineAxis } from '@/composables/d3/charts/heatmap';
import {
  daysOfMonths,
  ticksOfDay,
  monthNameShort,
  clockNames24,
} from '@/composables/utils/useDatetime';

export function weatherExtension(
  {
    svg = d3.create('svg'),
    // eslint-disable-next-line no-unused-vars
    width = 400,
    // eslint-disable-next-line no-unused-vars
    height = 900,
    // eslint-disable-next-line no-unused-vars
    margin = {
      top: 10,
      right: 30,
      bottom: 10,
      left: 30,
    },
  } = {},
  data,
) {
  if (data.length === 0) {
    return svg;
  }

  const { x, y } = defineAxis({
    xDomain: [0, data[0].length + 2],
    yDomain: [0, data.length + 2],
    ...arguments[0],
  });

  let color_weather = (d) => d3.interpolateBlues(d);
  let color_scale = d3.scaleBand().domain(weatherTypes).range([0.1, 1]);
  let color = (d) => color_weather(color_scale(d));
  let width_rect = x(2) - x(1) - 1;
  let height_rect = y(2) - y(1) - 1;
  /*  let tip = svg
    .append('div')
    .style('visibility', 'hidden')
    .style('background', 'white')
    .attr('font-size', '20pt')
    .attr('id', 'tip'); */
  svg
    .append('g')
    .attr('class', 'weather')
    .append('g')
    .attr('class', 'weather_upper_all')
    .selectAll('.weather_upper')
    .data(weatherHaikou)
    .join('rect')
    .attr('class', 'weather_upper')
    .attr('id', (d) => `weather_u_${d.days}`)
    .attr('x', (d, i) => x(1) + (x(2) - x(1)) * i)
    .attr('y', () => y(0))
    .attr('width', width_rect)
    .attr('height', height_rect)
    .attr('fill', (d) => color(d.weather_upper))
    .on('mouseover', (event, datum) => {
      svg
        .append('text')
        .attr('class', 'tip')
        .text(datum.weather_upper)
        .attr('text-anchor', 'middle')
        .attr('x', x(1) + (x(2) - x(1)) * datum.days)
        .attr('y', y(0) - 5)
        .attr('font-family', 'Verdana');
    })
    .on('mouseout', (event, datum) => {
      svg.selectAll('.tip').remove();
    });
  svg
    .select('.weather')
    .append('g')
    .attr('class', 'weather_lower_all')
    .selectAll('.weather_lower')
    .data(weatherHaikou)
    .join('rect')
    .attr('class', 'weather_lower')
    .attr('x', (d, i) => x(1) + (x(2) - x(1)) * i)
    .attr('y', () => y(data.length + 1))
    .attr('width', width_rect)
    .attr('height', height_rect)
    .attr('fill', (d) => color(d.weather_lower))
    .on('mouseover', (event, datum) => {
      svg
        .append('text')
        .attr('class', 'tip')
        .text(datum.weather_lower)
        .attr('text-anchor', 'middle')
        .attr('x', x(1) + (x(2) - x(1)) * datum.days)
        .attr('y', y(data.length + 1) + 21)
        .attr('font-family', 'Verdana');
    })
    .on('mouseout', (event, datum) => {
      svg.selectAll('.tip').remove();
    });

  return svg;
}

export function differentialExtension(
  {
    svg = d3.create('svg'),
    width = 400,
    height = 900,
    margin = {
      top: 10,
      right: 30,
      bottom: 10,
      left: 30,
    },
    color = ['#2196F3', '#FFC107'],
    scale = {
      vertical: 0.75,
      horizontal: 0.5,
    },
    fillOpacity = 0.8,
  },
  data,
) {
  if (data.length === 0) {
    return svg;
  }

  svg.select('.day_num_all').remove();
  svg.select('.hour_num_all').remove();

  const { x, y } = defineAxis({
    xDomain: [0, data[0].length + 2],
    yDomain: [0, data.length + 2],
    ...arguments[0],
  });

  let data_hour = [];
  let data_day = [];
  data.forEach((d) => {
    data_hour.push(d3.sum(d));
  });
  for (let i = 0; i < data[0].length; i++) {
    data_day.push(0);
    for (let j = 0; j < 24; j++) {
      data_day[i] += data[j][i];
    }
  }

  // Horizontal bar chart
  (() => {
    let data_day_slope = [];
    data_day_slope.push(0);
    for (let i = 1; i < data_day.length; i++) {
      data_day_slope.push(data_day[i] - data_day[i - 1]);
    }
    let day_x_scale = d3
      .scaleLinear()
      .domain([0, data[0].length + 2])
      .range([margin.left, width - margin.right]);
    let x_minus = day_x_scale(1) - day_x_scale(0) - 1;
    let day_y_scale_day = d3
      .scaleLinear()
      .domain([0, d3.max(data_day)])
      .range([height - margin.bottom, height]);
    svg
      .append('g')
      .attr('class', 'day_num_all')
      .selectAll('.day_num')
      .data(data_day)
      .join('rect')
      .attr('class', 'day_num')
      .attr('y', () => height - margin.bottom)
      .attr('x', (d, i) => {
        return x(1) + (x(2) - x(1)) * i;
      })
      .attr(
        'height',
        (d) => scale.horizontal * (day_y_scale_day(d) - day_y_scale_day(0)),
      )
      .attr('width', x_minus)
      .attr('fill', (d, i) => {
        return color[data_day_slope[i] >= 0 ? 0 : 1];
      })
      .attr('fill-opacity', fillOpacity);
  })();

  // Vertical bar chart
  (() => {
    let data_hour_slope = [];
    data_hour_slope.push(0);
    for (let i = 1; i < data_hour.length; i++) {
      data_hour_slope.push(data_hour[i] - data_hour[i - 1]);
    }
    let hour_x_scale_day = d3
      .scaleLinear()
      .domain(d3.extent(data_hour))
      .range([width - margin.right, width - 10]);
    let y_minus = y(2) - y(1) - 1;
    svg
      .append('g')
      .attr('class', 'hour_num_all')
      .selectAll('.hour_num')
      .data(data_hour)
      .join('rect')
      .attr('class', 'hour_num')
      .attr('x', () => width - margin.right)
      .attr('y', (d, i) => {
        return y(1) + (y(2) - y(1)) * i;
      })
      .attr('height', y_minus)
      .attr(
        'width',
        (d) => scale.vertical * (hour_x_scale_day(d) - hour_x_scale_day(0)),
      )
      .attr('fill', (d, i) => {
        return color[data_hour_slope[i] >= 0 ? 0 : 1];
      })
      .attr('fill-opacity', fillOpacity);
  })();
  return svg;
}

export function axisExtension(
  {
    svg = d3.create('svg'),
    margin = {
      top: 10,
      right: 30,
      bottom: 10,
      left: 30,
    },
  },
  data,
) {
  if (data.length === 0) {
    return svg;
  }

  const { x, y } = defineAxis({
    xDomain: [0, data[0].length + 2],
    yDomain: [0, data.length + 2],
    ...arguments[0],
  });
  const daysSum = daysOfMonths.map((value, index, array) => {
    if (index == 0) {
      return 0;
    } else {
      return array.slice(0, index).reduce((a, b) => a + b);
    }
  });

  svg
    .append('g')
    .attr('class', 'axis_month')
    .selectAll('.months_axis')
    .data(daysSum)
    .join('text')
    .attr('class', 'months_axis')
    .attr('x', (d) => x(1) + (x(2) - x(1)) * d)
    .attr('y', margin.top - 3)
    .text((d, i) => `${monthNameShort[i + 4]}`);

  svg
    .append('g')
    .attr('class', 'axis_hours')
    .selectAll('.hours_axis')
    .data(ticksOfDay)
    .join('text')
    .attr('class', 'hours_axis')
    .attr('x', x(1))
    .attr('text-anchor', 'end')
    .attr('y', (d) => y(1) + (y(2) - y(1)) * d)
    .text((d) => `${clockNames24[d / 3]}`);

  return svg;
}

export function datetimeExtension({
  svg = d3.create('svg'),
  width = 400,
  margin = {
    top: 10,
    right: 30,
    bottom: 10,
    left: 30,
  },
}) {
  let text = svg.append('g').attr('class', 'range');

  width = width + margin.left + margin.right;
  text
    .append('text')
    .attr('id', 'day_up')
    .attr('x', width / 2 - 5)
    .attr('y', 15)
    .text('May 1-Oct 31')
    .attr('text-anchor', 'end');

  text
    .append('text')
    .attr('id', 'hour')
    .attr('x', width / 2 + 5)
    .attr('y', 15)
    .text('0:00-24:00')
    .attr('text-anchor', 'start');

  return svg;
}
