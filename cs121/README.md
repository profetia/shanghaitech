# Cuckoo Hash

This repository contains the implementation of cuckoo hash in CUDA. Cuckoo hash is a scheme in computer programming for resolving hash collisions of values of hash functions in a table, with worst-case constant lookup time. The name derives from the behavior of some species of cuckoo, where the cuckoo chick pushes the other eggs or young out of the nest when it hatches in a variation of the behavior referred to as brood parasitism; analogously, inserting a new key into a cuckoo hash table may push an older key to a different location in the table.


## Build

To build the project, run the following commands:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
```

## Test

To run the tests, run the following command:

```bash
ctest -C Release
```

## Benchmark

The benchmark is done on a GeForce RTX 3070 GPU, the results are shown below:

![Performance Comparison](./assets/compare.svg)

![Insertion Performance](./assets/insert.svg)

![Lookup Performance](./assets/lookup.svg)