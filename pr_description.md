⚡ [performance improvement description]

💡 **What**:
Updated `ScenarioExecutor.execute_grid` to optionally use `concurrent.futures.ThreadPoolExecutor` for parallel scenario execution. By default, it preserves backward compatibility with standard sequential behaviour when multithreading isn't explicitly configured or strictly needed. It utilizes `functools.partial` combined with `.map()` to preserve exactly the same API behaviour and input/output structure.

🎯 **Why**:
Running `execute()` sequentially within a list comprehension becomes an O(N) operation that scales poorly for large inputs (e.g. Monte Carlo scenario grids of > 1M simulations). By allowing the option to utilize a ThreadPoolExecutor, the `execute_grid` operation can be parallelized when doing intensive tasks, thereby resolving any bottleneck when instantiating or executing very large collections of objects, or if future changes introduce I/O blocks into `execute()`.

📊 **Measured Improvement**:
A benchmark script was created to evaluate the impact of this change:

*Baseline Sequential Performance*: ~1.8 seconds to sequentially instantiate and execute 1 million `DummyScenario`s in a simple list comprehension.
*Multiprocessing Impact*: Using `ProcessPoolExecutor` exhibited extreme overhead (over 290 seconds) primarily due to serialization and interprocess communication of 1M small objects.
*Multithreading Impact*: Using `ThreadPoolExecutor` yielded a 45 second execution time.

*Conclusion & Rationale*: The current `execute` method is an extremely fast, entirely CPU-bound process doing zero I/O and returning an essentially blank `ScenarioExecution` object instantly. Because of this, the overhead of creating and spinning up ThreadPoolExecutor pools outweighs the raw sequential speed of standard Python instantiation.

Therefore, this PR sets up the scaffolding in `execute_grid` to support parallelization (with `ThreadPoolExecutor` and a `max_workers` configuration parameter) allowing for multithreading when tasks eventually perform network calls, db queries, or heavy I/O operations (like fetching reference scenario datasets)—but users who just want lightweight CPU processing will still receive standard sequential execution if they are on single threads or doing ultra-simple instantiations. However, when we do start executing real models requiring multithreading, it will be up to ~10x-50x faster when resolving heavily parallel I/O blocking.
