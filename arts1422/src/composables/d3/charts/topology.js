import * as d3 from 'd3';
import { getHaikouByCode } from '@/composables/utils/useHaikou';
import { doSharedDebounce } from '@/composables/utils/useDebounce';

export function topoGraph({
  data,
  width = 400,
  height = 900,
  margin = {
    top: 20,
    right: 30,
    bottom: 40,
    left: 30,
  },
  selectCallback = () => {},
  unselectCallback = () => {},
} = {}) {
  // console.log('data');
  // console.log(data);
  if (!data || data.length === 0) {
    return d3.create('svg');
  }
  let node_or = [];
  data.edges.forEach((d) => {
    // console.log('k');
    let x = d.x;
    let y = d.y;
    if (node_or.indexOf(x) == -1) {
      node_or.push(d.x);
    }
    if (node_or.indexOf(y) == -1) {
      node_or.push(d.y);
    }
  });

  // process data we obtained
  let nodes = [];
  node_or.forEach((element) => {
    nodes.push({
      id: element,
      in_number: d3.sum(data.edges, (d) => {
        if (d.x == element) {
          return d.yx_num;
        }
        if (d.y == element) {
          return d.xy_num;
        }
      }),
      out_number: d3.sum(data.edges, (d) => {
        if (d.x == element) {
          return d.xy_num;
        }
        if (d.y == element) {
          return d.yx_num;
        }
      }),
      neigebour: [element],
      group: null, //provided by clustering
    });
  });
  nodes.forEach((d) => {
    d.sum_number = d.in_number + d.out_number;
  });
  // console.log('node');
  // console.log(nodes);

  let node_weight_scale = d3
    .scaleLinear()
    .domain(d3.extent(nodes, (d) => d.sum_number))
    .range([1, 0]);
  nodes.forEach((d) => {
    d.weight = node_weight_scale(d.sum_number);
  });
  let links = [];
  for (let i = 0; i < data.edges.length; i++) {
    if (data.edges[i].x == data.edges[i].y) {
      // console.log('error on data for Topo');
      continue;
    }

    /*    links.push({
      source: weight ? data.edges[i].x : data.edges[i].y,
      target: weight ? data.edges[i].y : data.edges[i].x,
      weight: Math.abs(weight),
    }); */ //work for linkLine
    nodes[
      // eslint-disable-next-line no-unused-vars
      nodes.findIndex(function (item, _index, _arr) {
        return item.id == data.edges[i].x;
      })
    ].neigebour.push(data.edges[i].y);
    nodes[
      // eslint-disable-next-line no-unused-vars
      nodes.findIndex(function (item, _index, _arr) {
        return item.id == data.edges[i].y;
      })
    ].neigebour.push(data.edges[i].x);
    links.push({
      source: data.edges[i].x,
      target: data.edges[i].y,
      weight: data.edges[i].xy_num,
    });

    links.push({
      source: data.edges[i].y,
      target: data.edges[i].x,
      weight: data.edges[i].yx_num,
    });

    // work for linkArc
  }
  // console.log('edge');
  // console.log(links);

  let color = (d) => d3.interpolateRdYlGn(d);
  let color_scale = d3
    .scaleLinear()
    .domain(d3.extent(links, (d) => d.weight))
    .range([1, 0]);
  links.forEach((d) => {
    d.weight_color = color_scale(d.weight);
  });

  const svg = d3
    .create('svg')
    .attr('viewBox', [0, 0, width, height])
    .attr('width', width)
    .attr('height', height);
  // .style('background', '#C0C0C0');

  svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

  let linkArc = (d) => {
    const r = Math.hypot(d.target.x - d.source.x, d.target.y - d.source.y);
    return `
        M${d.source.x},${d.source.y}
        A${r},${r} 0 0,1 ${d.target.x},${d.target.y}
        `;
  };
  // eslint-disable-next-line no-unused-vars
  let linkLine = (d) => {
    const path = d3.path();
    path.moveTo(d.source.x, d.source.y);
    path.lineTo(d.target.x, d.target.y);
    return path.toString();
  };

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      'link',
      d3.forceLink(links).id((d) => d.id),
    )
    .force('charge', d3.forceManyBody().strength(-1600))
    .force('center', d3.forceCenter(width / 2, height / 2).strength(0.1));
  simulation.on('tick', () => {
    link.attr('d', linkArc);
    node.attr('transform', (d) => `translate(${d.x},${d.y})`);
    if (simulation.alpha() <= 0.05) {
      simulation.stop();
    }
  });
  // .on('end', ticked);

  let drag = (simulation) => {
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0.1);
      d.fx = null;
      d.fy = null;
      // console.log(event, d);
    }

    return d3
      .drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  };
  // initialize
  svg
    .selectAll('defs')
    .data(links)
    .join('defs')
    .append('marker')
    .attr('id', (d) => `Topo_arrow_${d.source.id}_${d.target.id}`)
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 15)
    .attr('refY', -0.5)
    .attr('markerWidth', 3)
    .attr('markerHeight', 5)
    .attr('fill', (d) => color(d.weight_color))

    .attr('orient', 'auto')
    .append('path')

    .attr('d', 'M0,-5L10,0L0,5');


  const [linesMouseover, linesMouseout] = doSharedDebounce([
    (event, datum) => {
      svg.selectAll('.tip').remove();
      svg
        .append('text')
        .attr('class', 'tip')
        .append('tspan')
        .text(`Source:${getHaikouByCode(datum.source.id).properties.name}`)
        .attr('text-anchor', 'start')
        .attr('x', event.layerX + 15)
        .attr('y', event.layerY - 15);
      svg
        .select('.tip')
        .append('tspan')
        .text(`Target:${getHaikouByCode(datum.target.id).properties.name}`)
        .attr('text-anchor', 'start')
        .attr('x', event.layerX + 15)
        .attr('y', event.layerY)
        .attr('font-family', 'Verdana');
      svg
        .select('.tip')
        .append('tspan')
        .text(`Num:${datum.weight}`)
        .attr('text-anchor', 'star')
        .attr('x', event.layerX + 15)
        .attr('y', event.layerY + 15)
        .attr('font-family', 'Verdana');
    }, 
    // eslint-disable-next-line no-unused-vars
    (_event, _datum) => {
      svg.selectAll('.tip').remove();
    }
  ], [200, 0]);

  const link = svg
    .append('g')
    .attr('fill', 'none')

    .selectAll('.lines')
    .data(links)
    .join('path')
    .attr(
      'class',
      (d) =>
        `lines Topo_line_target_${d.target.id} Topo_line_source_${d.source.id}`,
    )
    .attr('id', (d) => `Topo_line_${d.source.id}_${d.target.id}`)
    .style('stroke', (d) => color(d.weight_color))
    .attr('stroke-width', 4)
    .attr('fill', 'none')
    .attr('isCalled', 'false')
    .attr(
      'marker-end',
      (d) =>
        `url(${new URL(
          `#Topo_arrow_${d.source.id}_${d.target.id}`,
          location,
        )})`,
    )
    .on('mouseover', linesMouseover)
    .on('mouseout', linesMouseout);


  const [nodeMouseover, nodeMouseout] = doSharedDebounce([
    (event, datum) => {
      svg.selectAll('.tip').remove();
      svg
        .append('text')
        .attr('class', 'tip')
        .append('tspan')
        .text(`Place:${getHaikouByCode(datum.id).properties.name}`)
        .attr('text-anchor', 'start')
        .attr('x', event.layerX + 15)
        .attr('y', event.layerY - 15);
      svg
        .select('.tip')
        .append('tspan')
        .text(`In:${datum.in_number}`)
        .attr('text-anchor', 'start')
        .attr('x', event.layerX + 15)
        .attr('y', event.layerY)
        .attr('font-family', 'Verdana');
      svg
        .select('.tip')
        .append('tspan')
        .text(`Out:${datum.out_number}`)
        .attr('text-anchor', 'star')
        .attr('x', event.layerX + 15)
        .attr('y', event.layerY + 15)
        .attr('font-family', 'Verdana');
    }, 
    // eslint-disable-next-line no-unused-vars
    (_event, _datum) => {
      svg.selectAll('.tip').remove();
    }
  ], [200, 0]);


  const node = svg
    .selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('class', 'nodes')
    .attr('id', (d) => `Topo_node_${d.id}`)
    .attr('r', 5)
    .style('fill', (d) => getHaikouByCode(d.id).properties.color)
    .attr('stroke-width', 2)
    .attr('stroke', '#666')
    .attr('isCalled', 'false')
    .call(drag(simulation))
    .on('click', function (d) {
      // console.log(
      //   d3
      //     .selectAll(`.Topo_line_target_${d.target.__data__.id}`)
      //     .attr('isCalled'),
      // );

      if (
        svg.selectAll(`#Topo_node_${d.target.__data__.id}`).attr('isCalled') ==
        'true'
      ) {
        unselectCallback(...arguments);
        svg
          .selectAll('.nodes')
          .attr('opacity', '1')
          .attr('isCalled', 'false')
          .attr('opacity', '1')
          .attr('r', 5)
          .attr('stroke', '#666')
          .attr('stroke-width', 2);
        svg.selectAll('.lines').attr('opacity', '1').attr('isCalled', 'false');
      } else {
        selectCallback(...arguments);
        d3.selectAll('.nodes').attr('opacity', '0.1').attr('isCalled', 'false');
        d3.selectAll('.lines').attr('opacity', '0.1').attr('isCalled', 'false');
        // console.log(d.target.__data__);
        d.target.__data__.neigebour.forEach((d) => {
          d3.selectAll(`#Topo_node_${d}`)
            .attr('opacity', '1')
            .attr('r', 5)
            .attr('stroke', '#666')
            .attr('stroke-width', 2);
        });
        d3.selectAll(`#Topo_node_${d.target.__data__.id}`)
          .attr('r', 8)
          .attr('stroke', 'white')
          .attr('stroke-width', 3)
          .attr('isCalled', 'true');
        d3.selectAll(`.Topo_line_target_${d.target.__data__.id}`)
          .attr('opacity', '1')
          .attr('isCalled', 'false');
        d3.selectAll(`.Topo_line_source_${d.target.__data__.id}`)
          .attr('opacity', '1')
          .attr('isCalled', 'false');
      }
    })
    .on('mouseover', nodeMouseover)
    .on('mouseout', nodeMouseout);

  return svg;
}
