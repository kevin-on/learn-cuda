## 2.2.3.4 Local Memory

### Local memory layout note

> "Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs."

This means local memory is laid out in a **thread-interleaved, word-granular** way.
For each 32-bit word position inside a variable, threads in a warp are placed next to each other.

Example: warp threads `t0..t3`, variable `V` has 3 words (`V.w0`, `V.w1`, `V.w2`).

| Local memory order | Stored word |
|---|---|
| 0 | `t0.V.w0` |
| 1 | `t1.V.w0` |
| 2 | `t2.V.w0` |
| 3 | `t3.V.w0` |
| 4 | `t0.V.w1` |
| 5 | `t1.V.w1` |
| 6 | `t2.V.w1` |
| 7 | `t3.V.w1` |
| 8 | `t0.V.w2` |
| 9 | `t1.V.w2` |
| 10 | `t2.V.w2` |
| 11 | `t3.V.w2` |

So for any word index `k`, accesses to `V.wk` by consecutive thread IDs are contiguous/coalescable.
