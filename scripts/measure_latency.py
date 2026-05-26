"""Measure per-step latency for a simple and a complex query."""
import sys
import time
sys.path.insert(0, ".")

from src.utils.logger import configure_logging
configure_logging()

from src.factory import get_rag_components

orchestrator, _ = get_rag_components()

# Warm-up
orchestrator._retriever.retrieve("test", top_k=1)


def measure(query: str) -> dict:
    steps = {}

    t = time.perf_counter()
    sub_queries = orchestrator._query_decomposer.decompose(query)
    steps["decomposer"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    chunks = orchestrator._retrieve_with_decomposition(query)
    steps["retrieval"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    orchestrator._generator.generate(query, chunks)
    steps["generation"] = (time.perf_counter() - t) * 1000

    steps["total"] = sum(steps.values())
    steps["sub_queries"] = len(sub_queries)
    steps["chunks"] = len(chunks)
    return steps


SIMPLE  = "What does Article 21 of the Indian Constitution protect?"
COMPLEX = "How do Fundamental Rights and Directive Principles differ in enforceability?"

print("Measuring simple query  (3 runs avg)...")
simple_runs  = [measure(SIMPLE)  for _ in range(3)]
print("Measuring complex query (3 runs avg)...")
complex_runs = [measure(COMPLEX) for _ in range(3)]


def avg(runs, key):
    return sum(r[key] for r in runs) / len(runs)


print()
print("=" * 60)
print(f"SIMPLE:  {SIMPLE}")
print(f"COMPLEX: {COMPLEX}")
print("=" * 60)
print(f"{'Step':<14} {'Simple':>10} {'Complex':>10}")
print("-" * 36)
for step in ("decomposer", "retrieval", "generation", "total"):
    print(f"{step:<14} {avg(simple_runs, step):>9.0f}ms {avg(complex_runs, step):>9.0f}ms")
print("-" * 36)
print(f"{'sub-queries':<14} {avg(simple_runs,'sub_queries'):>10.1f} {avg(complex_runs,'sub_queries'):>10.1f}")
print(f"{'chunks':<14} {avg(simple_runs,'chunks'):>10.1f} {avg(complex_runs,'chunks'):>10.1f}")
