# Chapter 2: Programming GPUs in CUDA

### Local memory layout note

> "Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs."

Intuition:
- Think of local memory as arranged by `(word offset first, thread ID second)`.
- For a given word offset `k`, threads `t0, t1, t2, ...` are placed next to each other.

Why this matters:
- If a warp accesses the same offset (for example all threads read `V.w1`), accesses are naturally contiguous and easier to coalesce.
- Local memory is still backed by device memory, so this helps access pattern quality, not raw latency.

### Warp Switching vs CPU Context Switching

- CPU switch: save one thread state and restore another, which is relatively expensive.
- GPU warp switch: many warp states are already resident on the SM, so scheduler picks another ready warp quickly.

Core idea:
- GPUs hide memory latency by switching to ready warps, not by making a single thread wait less.
- Large on-chip register files are what make this possible.

### Register Spill vs Launch Failure

These are different failure/performance modes:

- **Register spill (compile time)**: compiler cannot keep all live values in registers, so some values go to local memory.
- **Launch failure (runtime)**: block resource request cannot fit on an SM (registers/shared memory/threads), so kernel config is invalid.

Quick mental model:
- Spill = performance problem.
- Launch failure = configuration problem.

### GPU Latency Cheat Sheet (Approximate)

Rough order (fast to slow):
- Register: ~1 cycle
- Shared memory / L1 hit: tens of cycles
- L2 hit: hundreds of cycles
- DRAM (global miss path): hundreds to 1000+ cycles

Important fundamentals:
- Exact numbers vary by architecture and contention.
- Throughput depends on keeping enough warps ready while long-latency accesses are in flight.

### L2 Slices and Crossbar/NoC Interconnect

- L2 is split into slices.
- SM requests reach those slices through one on-chip interconnect.
- Different GPUs implement that interconnect differently (crossbar or NoC).
- Any SM can usually reach any slice; address mapping is interleaved/hashed.

Performance implication:
- Contention and uneven partition usage are usually bigger bottlenecks than physical distance.

### Kernel Launch Latency

- Every kernel launch has fixed overhead before math begins (runtime, driver, GPU front-end work).
- For large kernels this is small, but for many tiny kernels it can dominate end-to-end latency.
- This is especially common in DL inference (for example, batch size `1` with many small ops).
- Example fix: use fused kernels such as FlashAttention, which reduce kernel-launch overhead (and often memory traffic) by combining multiple attention steps.

#### Open Question (Not Fully Resolved Yet)

**Q: What are the main sources of CUDA kernel launch latency?**
