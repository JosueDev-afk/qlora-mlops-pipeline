"""Model servers: Qwen through vLLM and Laya with its router preloaded.

A third zone with its own dependency group (`serve`). torch lives here and never
in `src/agent/`, which reaches both models over HTTP. Serving both models the
same way on the same reference GPU is also what makes H1's latency comparison
fair: neither model skips the network hop the other pays.
"""
