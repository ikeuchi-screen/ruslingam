---
outline: deep
---

# Threading

The causal-order search and the `p(p-1)/2` error-independence HSIC tests run in
parallel on a [Rayon](https://docs.rs/rayon) thread pool. By default this is the
global pool: one worker per logical core, or `RAYON_NUM_THREADS` if it is set in
the environment before the first `fit()`.

You can also control it from Python at any time — including after `fit()` has run,
because `set_num_threads` uses a dedicated pool rather than rayon's build-once
global one:

```python
import ruslingam

ruslingam.set_num_threads(4)   # run the search on a dedicated pool of 4 threads
ruslingam.set_num_threads(0)   # back to the default (all cores / RAYON_NUM_THREADS)
ruslingam.get_num_threads()    # -> worker count currently in effect
```

Results (`causal_order_`, `adjacency_matrix_`, and the bootstrap aggregates) are
identical regardless of the thread count.

See [`set_num_threads` / `get_num_threads`](/module-functions).
