"""Code shared by both zones.

Only the base dependencies in `[project].dependencies` may be imported here, so
that `src/agent/` can use it without pulling in Spark or torch.
"""
