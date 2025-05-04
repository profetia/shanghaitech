# API Design

## Heatmap

GET /heatmap

### Request

```javascript
{
  // empty
}
```

### Response

heatmap as an array with shape of 150x24 and initial geometries hotspots.

```javascript
{
  "init_geometry": [
    {
      "id": 46010701, // int, unique id of node
      // <geo_json_id><num_node_id>
      // geo_json_id: 460107
      // num_node_id: 01
      // this means the node is the 01th node in geo_json 460107
      "lng": 20.00
      "lat": 110.11
      "start_num": 234, // int
      "dest_num": 567 // int
    },
    ...
  ]
  "heatmap": [ // The first dim indicate the date id
    [200, 0, 123, ..., 123], // the second dim indicate the clock id at that date
    [200, 0, 123, ..., 123],
    [200, 0, 123, ..., 123],
    ...
]
}
```

## Geometry

POST /geometry

Get the map view data.

### Request

```javascript
{
  "sample": true, // whether to sample the data
  "date_range": [1, 133], // 0 ~ 200
  "clock_range": [0, 21] // 0 ~ 23
}
```

### Response

An array of nodes.

```javascript
[
  {
    "id": 46010701, // int, unique id of node
    // <geo_json_id><num_node_id>
    // geo_json_id: 460107
    // num_node_id: 01
    // this means the node is the 01th node in geo_json 460107
    "lng": 20.00
    "lat": 110.11
    "start_num": 234, // int
    "dest_num": 567 // int
  },
  ...
]
```

## Topology

POST /topology

### Request

```javascript
{
  "nodes": [46010701, 46010702],
  // chosen nodes
  // an array of int, unique id of nodes
  "date_range": [1, 3],
  "clock_range": [3, 12]
}
```

### Response

```javascript
{
  "nodes": [46010701, 46010702, 46010723, 46010708, 46010709],
  // all nodes include the chosen and the unchosen
  "edges": [
    {
      "x": 46010702,
      "y": 46010708,
      "xy_num": 32, // order num of x to y
      "yx_num": 47 //
    },
    ...
  ]
}
```
