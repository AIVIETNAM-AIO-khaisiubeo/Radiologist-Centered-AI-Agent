"""Agent loop with guardrails and a full trace."""
from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone

from radagent.agent.planner import RulePlanner
from radagent.agent.tools import TOOLS, CaseContext, ready_tools, requirements
from radagent.store import dump_json


def validate(proposal: dict, ctx: CaseContext) -> str | None:
    """Return None if the proposal may run, otherwise the reason it is refused."""
    tool = proposal.get("tool")
    if proposal.get("error"):
        return f"planner error: {proposal['error']}"
    if tool not in TOOLS:
        return f"unknown tool {tool!r}"
    if tool in ctx.done:
        return f"{tool} already completed"
    if tool == "apply_reader_policy" and ctx.reader_policy is None:
        return "no reader assigned"
    missing = [r for r in requirements(tool, ctx) if r not in ctx.done]
    if missing:
        return f"{tool} requires {missing}"
    if proposal.get("args"):
        return "tools take no free arguments"
    return None


def run_agent(ctx: CaseContext, planner=None, max_steps: int = 12, save: bool = True) -> dict:
    planner = planner or RulePlanner()
    fallback = RulePlanner()
    trace = {"case_id": ctx.case_id, "planner": getattr(planner, "name", "rule"),
             "planner_model": getattr(planner, "model", None),
             "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "steps": [], "status": "running"}
    for step in range(1, max_steps + 1):
        t0 = time.time()
        proposal = planner.propose(ctx)
        plan_ms = (time.time() - t0) * 1000
        refused = validate(proposal, ctx)
        source = trace["planner"]
        if refused:
            proposal = fallback.propose(ctx)
            source = "rule (guardrail fallback)"
        tool = proposal["tool"]
        t1 = time.time()
        entry = {"step": step, "tool": tool, "planned_by": source, "reason": proposal.get("reason", ""),
                 "guardrail": refused, "planning_ms": round(plan_ms, 1), "ready_before": ready_tools(ctx)}
        try:
            summary = TOOLS[tool].fn(ctx)
            entry.update({"status": "ok", "summary": summary})
            ctx.done[tool] = summary
        except Exception as e:  # a failed tool stops the read and routes the case to a human
            entry.update({"status": "error", "summary": {"error": f"{type(e).__name__}: {e}"},
                          "traceback": traceback.format_exc(limit=3)})
            trace["steps"].append({**entry, "duration_ms": round((time.time() - t1) * 1000, 1)})
            trace["status"] = "error"
            break
        entry["duration_ms"] = round((time.time() - t1) * 1000, 1)
        trace["steps"].append(entry)
        if tool == "finish":
            trace["status"] = "complete"
            break
    else:
        trace["status"] = "max_steps"
    trace["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trace["n_guardrail_overrides"] = sum(1 for s in trace["steps"] if s.get("guardrail"))
    ctx.record["agent"] = {"planner": trace["planner"], "status": trace["status"], "n_steps": len(trace["steps"]),
                           "guardrail_overrides": trace["n_guardrail_overrides"]}
    if save:
        dump_json(trace, ctx.store.trace_path(ctx.case_id))
    return trace
