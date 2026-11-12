# pocket-vectors

Exact, on-device vector search for iOS 27 and macOS 27, in Swift, with no dependencies beyond the OS.

- **`PocketVectors`**: a flat (brute-force) index of int8, fp16 or fp32 vectors. NEON kernels (the
  ARMv8.2 `sdot` instruction for int8), memory-mapped single-file storage, row filters, `Sendable`.
- **`PocketVectorsCoreML`** (optional): a Core ML sentence-embedding encoder (BERT-family models such as
  bge-small or MiniLM) with a matching WordPiece tokenizer, pinned to the **Neural Engine**, plus
  `ComputePlanReport` to check that the model really runs there.

For a few hundred thousand vectors or fewer, an exact scan of int8 vectors is as fast as a graph
index, smaller, simpler, and filters exactly. That is the whole idea.

## When flat beats HNSW

A flat int8 scan reads `rows × dim` bytes per query: 15 MB for 40k × 384, which an Apple-silicon core
streams in a fraction of a millisecond. HNSW reads far less per query but:

- **it is approximate**: matching flat int8's recall needs a wide search (high `ef`), which costs
  most of its speed advantage at this size;
- **it is bigger**: the graph adds links on top of the vectors, and loading it (rather than mapping it)
  puts it all in the app's memory footprint;
- **filtered search degrades**: with a selective filter (one collection, one date range) a graph
  search has to walk past most of what it finds, and gets slow or loses recall. A flat scan costs the
  same with any filter, and a contiguous row range scans only that slice.

Reach for an approximate index when a single index holds millions of vectors, or when the scan no
longer fits your latency budget on the slowest device you support. Measure first.

## Measured

Same 38,482 × 384 text embeddings (bge-small-en-v1.5, unit length), the same 500 queries, recall@10
against exact fp32 search, natively on an M3 Max (macOS 27.2.1), one query at a time:

| Index | recall@10 | p50 / p95 ms | on disk | filter to 539 contiguous rows, p95 | added code |
|---|---:|---:|---:|---:|---:|
| **pocket-vectors int8** (memory-mapped) | **0.978** | **0.29 / 1.24** | **15.1 MB** | 1.1 ms (masked full scan; a row range scans only the slice: see the synthetic bench) | ~2 KB C + Swift |
| USearch 2.26.6 HNSW int8, M 16, ef 128, memory-mapped | 0.966 | 0.12 / 0.17 | 20.6 MB | 8.5 ms | 1.2 MB |
| USearch 2.26.6 HNSW fp16, M 16, ef 128 | 0.982 | 0.37 / 0.56 | 35.4 MB (+45 MB resident) | 9.1 ms | 1.2 MB |
| USearch 2.26.6 exact search, int8 | 0.981 | 0.73 / 0.81 | 15.1 MB | 0.014 ms (range) | 1.2 MB |
| sqlite-vector 1.2.0, int8 full scan | 0.981 | 2.9 / 6.0 | 16.5 MB | 11.2 ms | 126 KB |

Measured on a busy machine (load average 600–700 on 16 cores), so absolute times are inflated;
recall, sizes and the ordering are not. On a phone, expect the int8 scan to be bound by memory
bandwidth: 15 MB per query.

Synthetic (`pocket-vectors-bench`, 40,000 × 384, 300 queries, same machine and load):

| format | file MB | open ms (mmap) | p50 ms | p95 ms | recall@10 |
|---|---:|---:|---:|---:|---:|
| float32 | 61.4 | 0.49 | 0.84 | 1.31 | 1.0000 |
| float16 | 30.7 | 0.14 | 0.75 | 0.91 | 0.9993 |
| int8 | 15.5 | 0.07 | 0.26 | 0.37 | 0.9753 |

Filters on int8, p50: a row range of 10 % of rows 0.028 ms; a `RowSet` of every 10th row 0.32 ms; a
predicate 0.32 ms.

Synthetic numbers you can reproduce anywhere: `swift run -c release pocket-vectors-bench` (seeded,
clustered unit vectors; no data download).

## Use

```swift
import PocketVectors

// Build once (offline or on device): unit-length vectors, dimension a multiple of 16.
try FlatIndex.write(vectors, format: .int8, to: url)

// Open: memory-mapped, O(1). Page it in off the main thread before the first query.
let index = try FlatIndex(contentsOf: url)
Task.detached { index.prefault() }

let hits = index.search(queryVector, k: 10)                       // [SearchResult(row:score:)]
let inGroup = index.search(queryVector, k: 10, filter: .rows(12_000..<15_400))   // scans only the slice
let tagged = RowSet(capacity: index.count, rows: rowsWithTag)      // build once per filter value
let some = index.search(queryVector, k: 10, filter: .rows(in: tagged))
```

Rows are numbered in the order you wrote them; keep your own table from row to document. Write rows
grouped by the field you filter on most, and that filter becomes a range.

With a Core ML encoder:

```swift
import PocketVectorsCoreML

var config = CoreMLTextEncoder.Configuration()          // computeUnits = .cpuAndNeuralEngine
config.prefix = "Represent this sentence for searching relevant passages: "   // bge's query prefix
let encoder = try CoreMLTextEncoder(modelURL: mlmodelcURL,
                                    tokenizer: WordPieceTokenizer(vocabURL: vocabURL),
                                    configuration: config)
let hits = index.search(try encoder.embed("quiet places to read on a rainy day"), k: 10)
```

The model takes `input_ids` (`Int32`, `[1, L]` for fixed lengths such as 32/64/128) and outputs the
pooled embedding; fold pooling and normalisation into the model when you convert it with coremltools.

## Proving the encoder runs on the Neural Engine

Asking for `.cpuAndNeuralEngine` is not proof. Three checks, cheapest first:

1. **The compute plan.** `try await ComputePlanReport.load(modelURL:computeUnits:)` lists, for every op,
   the device Core ML dispatches it to. For bge-small (int8 weights) on an M3 Max: 635 of 658 ops on
   the Neural Engine, 88 % of the estimated cost; the rest are casts, gathers and shape ops on the CPU.
   With `.all`, the same model was planned **entirely on the GPU** on a Mac, which is why the default
   here is `.cpuAndNeuralEngine`.
2. **Timing.** The same queries with `.cpuAndNeuralEngine` and `.cpuOnly`; on a quiet machine the ANE
   run is several times faster.
3. **A trace.** Instruments' "Neural Engine" instrument (`xcrun xctrace record --instrument 'Neural
   Engine' --launch -- your-tool`) shows one "Neural Engine Prediction" interval per query.

The iOS simulator has no Neural Engine; check on a device.

## Requirements

iOS 27 / macOS 27, Swift 6. Dimension a multiple of 16; `k` up to 1,024.

## Licence

MIT. See [LICENSE](LICENSE).
