import * as d3 from 'd3';

export class D3Proxy {
  #index = null;
  #node = null;
  constructor(data) {
    Object.assign(this, data);
  }

  getD3Node() {
    return this.#node;
  }

  setD3Node(value) {
    this.#node = value;
  }

  getD3Index() {
    return this.#index;
  }

  setD3Index(value) {
    this.#index = value;
  }

  unwrap() {
    return {
      ...this,
    };
  }
}

export function d3Proxy(target) {
  return new D3Proxy(target);
}

export function d3ProxyArray(target) {
  return target.map((d) => d3Proxy(d));
}

export function d3ProxyFunctionWrapper(callback) {
  return function (datum, index, group) {
    datum.setD3Index(index);
    datum.setD3Node(group[index]);
    return callback.apply(this, [datum, index, group]);
  };
}

export function d3ProxyBindNodes(selection) {
  selection.attr(
    'd3-proxy',
    d3ProxyFunctionWrapper(() => null),
  );
}

export function d3ProxyGroupNodes(
  selection,
  formatter = (d, i, g) => `d3-proxy-${i}`,
) {
  selection.classed('', (d, i, g) => {
    const node = d3.select(g[i]);
    node.classed(formatter(d, i, g), true);
  });
}

/*
@depreciated: Using class implement now.
@since: 2022-11-06

export class D3Proxy {
  constructor(data) {
    this.data = data;
    this.node = null;
    this.index = null;
  }
}

export function d3Proxy(target) {
  return new Proxy(new D3Proxy(target), {
    get(target, prop) {
      if (prop == 'getD3Node') {
        return () => target.node;
      } else if (prop == 'setD3Node') {
        return (node) => (target.node = node);
      } else if (prop == 'getD3Index') {
        return () => target.index;
      } else if (prop == 'setD3Index') {
        return (index) => (target.index = index);
      } else if (prop == 'unwrap') {
        return () => target.data;
      } else {
        return target.data[prop];
      }
    },

    set(target, prop, value) {
      if (prop == 'getD3Node' || prop == 'setD3Node') {
        return false;
      } else {
        target.data[prop] = value;
        return true;
      }
    },
  });
}
*/
