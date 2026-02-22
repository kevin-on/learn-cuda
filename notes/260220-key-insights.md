## CPU vs GPU vs TPU vs NPU — Key Insights

**1. Everything starts from a single trade-off.**
Given a fixed transistor budget, you can spend it on "making a single thread fast" or "running many threads at once." CPU chose the former, GPU chose the latter.

**2. Every GPU design decision rests on one premise.**
"There exists a massive amount of independent, identical work." If this premise breaks, a GPU is just a slow, simple CPU. SIMT, warp scheduling, massive register files — none of it makes sense without this premise.

**3. Reducing latency vs hiding latency.**
CPU reduces latency through caches, branch prediction, and out-of-order execution. GPU hides latency by keeping thousands of threads resident and switching between them. Completely different strategies, but both aim at the same goal: keep the ALUs busy.

**4. GPU cache being slow is not a flaw — it's by design.**
GPU L1 taking ~30 cycles isn't due to lack of optimization. It's the cost of arbitration logic when hundreds of ALUs hit the same cache simultaneously, and there's no incentive to reduce it — they'll just hide it. Logic overhead dominates over physical distance (8:2 or more).

**5. GPU registers and CPU registers are physically different things.**
CPU builds a small number from expensive flip-flops. GPU builds a massive amount from SRAM. A 256 KB register file is possible because the area per bit is 3–6x cheaper. But the access mechanism differs (index lookup vs direct wiring), so despite sharing the same name, they differ in speed.

**6. Warp size 32 is not a theoretical optimum — it's a practical equilibrium.**
Larger warps improve control logic sharing but increase branch divergence penalties. AMD's move from 64 to 32 proves this. AI-specific chips with no branching go much wider.

**7. Specialization means cutting what you don't need.**
CPU → GPU: Cut complex control logic, fill with ALUs.
GPU → TPU: Cut SIMT, warp scheduling, general-purpose cores; replace with a systolic array dedicated to matrix multiplication.
TPU → Edge NPU: Cut training, high precision; specialize for INT4/INT8 inference.
Each step is the result of the conviction: "our workload doesn't need this."

**8. Why TPU doesn't crush GPU despite being purpose-built.**
A dedicated design should theoretically be far more efficient, but the actual gap is only 1.2–2x. NVIDIA already added Tensor Cores (essentially small systolic arrays), the real bottleneck is memory bandwidth rather than compute, and CUDA's 15+ years of software optimization closes much of the remaining gap.

**9. The true cost of modern AI is data movement, not computation.**
This is the insight Tenstorrent exploits. In a GPU, all SMs share a single HBM — adding more cores doesn't increase available bandwidth. Tenstorrent distributes memory (1.5 MB SRAM per core), giving each core its own local storage and using an explicit network (NoC) for data transfer. More cores means more total bandwidth. The trade-off is programming difficulty.

**10. How well you understand your workload determines your architecture.**
GPU: "We don't know what's coming" → general-purpose parallel processor.
TPU: "We only need matrix multiplication" → systolic array.
Tenstorrent: "Data movement is the bottleneck" → distributed SRAM + NoC.
The stronger your conviction about the workload, the more you can cut away — and the more you cut, the more efficient you become.