// swift-tools-version:6.0
// pocket-vectors: exact on-device vector search for Apple platforms. A flat int8 / fp16 /
// fp32 index with NEON kernels, memory-mapped files and row filters (PocketVectors), plus an optional
// Core ML text encoder pinned to the Neural Engine and a compute-plan check (PocketVectorsCoreML).
// No dependencies beyond the OS. MIT licence.
import PackageDescription

let package = Package(
    name: "pocket-vectors",
    platforms: [.iOS("27.0"), .macOS("27.0")],
    products: [
        .library(name: "PocketVectors", targets: ["PocketVectors"]),
        .library(name: "PocketVectorsCoreML", targets: ["PocketVectorsCoreML"]),
        .executable(name: "pocket-vectors-bench", targets: ["pocket-vectors-bench"]),
    ],
    targets: [
        .target(name: "CPocketVectors", cSettings: [.define("NDEBUG")]),
        .target(name: "PocketVectors", dependencies: ["CPocketVectors"]),
        .target(name: "PocketVectorsCoreML"),
        .executableTarget(name: "pocket-vectors-bench", dependencies: ["PocketVectors"]),
        .testTarget(name: "PocketVectorsTests", dependencies: ["PocketVectors", "CPocketVectors"]),
        .testTarget(name: "PocketVectorsCoreMLTests", dependencies: ["PocketVectorsCoreML"]),
    ],
    swiftLanguageModes: [.v6]
)
