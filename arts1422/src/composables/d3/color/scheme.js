import * as d3 from 'd3';

export function discreteScheme(category, max, policy = d3.interpolateRainbow) {
  return [...Array(category)].map((_, i) => {
    const upper = (max / category) * (i + 1);
    return policy(upper / max);
  });
}
