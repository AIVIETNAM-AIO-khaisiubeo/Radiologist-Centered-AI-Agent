"""Planners choose the next tool; the runner decides whether that choice is allowed.

``RulePlanner``  deterministic, explainable default (no model needed).
``LLMPlanner``   an open LLM (OpenAI-compatible endpoint) proposes the next tool as
                 JSON. It sees tool *descriptions and summaries of previous outputs*
                 only - never image data and never free text it could copy into a
                 report - and every proposal is validated by the runner. Invalid or
                 failed proposals fall back to the rule planner and are logged.
"""
from __future__ import annotations

import json

from radagent.agent.llm_client import ChatClient, LLMError, extract_json
from radagent.agent.tools import TOOLS, CaseContext, ready_tools, requirements

ORDER = ["load_study", "segment_liver_tumour", "measure_findings", "quality_check", "vlm_second_read",
         "estimate_confidence",
         "triage_case", "apply_reader_policy", "draft_report", "finish"]


class RulePlanner:
    name = "rule"

    def propose(self, ctx: CaseContext) -> dict:
        ready = ready_tools(ctx)
        for tool in ORDER:
            if tool in ready:
                return {"tool": tool, "args": {}, "reason": self._reason(tool, ctx)}
        return {"tool": "finish", "args": {}, "reason": "nothing left to do"}

    @staticmethod
    def _reason(tool: str, ctx: CaseContext) -> str:
        qc = ctx.done.get("quality_check", {})
        if tool == "estimate_confidence" and qc and not qc.get("passed", True):
            return "QC failed - still estimate confidence for the record, but the case will go to a manual read"
        if tool == "triage_case" and qc and not qc.get("passed", True):
            return "QC failed - triage as QC FAIL (manual read)"
        return {
            "load_study": "start: read the CT and its geometry",
            "segment_liver_tumour": "geometry known - segment liver and tumour",
            "measure_findings": "masks available - measure volumes from voxel spacing",
            "quality_check": "measurements available - check plausibility before trusting them",
            "vlm_second_read": "measurements done - get an independent second read from the vision-language model",
            "estimate_confidence": "combine segmenter and second read into one calibrated probability",
            "triage_case": "confidence and QC available - prioritise the case",
            "apply_reader_policy": "a reader is assigned - apply their learned suggestion/deferral policy",
            "draft_report": "fill the structured report with measured values",
            "finish": "report drafted",
        }.get(tool, "")


SYSTEM_PROMPT = """You are the planning module of a research radiology agent that performs a first read of liver CT.
You choose which tool to run next. You never describe images, never write findings and never invent values:
tools compute everything, the report is a template filled by a tool, and the vision-language tool's own text
is quoted there as unverified.
Rules: run a tool only if all its required tools have completed; run each tool at most once;
if quality control failed, the case must still be triaged (it will be marked for a manual read) and reported;
call "finish" only after "draft_report".
Respond with ONE JSON object and nothing else: {"tool": "<name>", "args": {}, "reason": "<one short sentence>"}"""


class LLMPlanner:
    name = "llm"

    def __init__(self, llm_cfg: dict):
        self.client = ChatClient(llm_cfg["base_url"], llm_cfg["model"], llm_cfg.get("api_key_env", "LLM_API_KEY"),
                                 llm_cfg.get("timeout_s", 60), llm_cfg.get("temperature", 0.0))
        self.model = llm_cfg["model"]

    def state(self, ctx: CaseContext) -> dict:
        tools = []
        for name, t in TOOLS.items():
            if name == "apply_reader_policy" and ctx.reader_policy is None:
                continue
            tools.append({"name": name, "description": t.description, "requires": requirements(name, ctx),
                          "completed": name in ctx.done})
        return {"case_id": ctx.case_id, "reader_assigned": ctx.reader_policy is not None,
                "completed_steps": [{"tool": k, "result": v} for k, v in ctx.done.items()], "tools": tools}

    def propose(self, ctx: CaseContext) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(self.state(ctx), default=str)}]
        try:
            out = extract_json(self.client.chat(messages))
        except LLMError as e:
            return {"tool": None, "args": {}, "reason": "", "error": str(e)}
        return {"tool": out.get("tool"), "args": out.get("args") or {}, "reason": str(out.get("reason", ""))[:300]}


def make_planner(cfg: dict):
    if cfg["agent"]["planner"] == "llm":
        return LLMPlanner(cfg["agent"]["llm"])
    return RulePlanner()
