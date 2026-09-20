# executor_runner.py

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Dict, Optional

from agent_framework import (
    WorkflowBuilder,
    WorkflowContext,
    executor,
    FileCheckpointStorage,
)

from telemetry.otel import tracer

from agents_main.classification_agent import build_agent as build_classification_agent
from agents_main.indexing_agent import _build_single_invoice_agent as build_indexing_agent

from tools.hitl import resolve_hitl_decision
from storage.workflow import WorkflowStore


logger = logging.getLogger(__name__)

MAX_INDEXING_HITL_AUDIT_RETRIES = 2
MAX_CLASSIFICATION_AUDIT_RETRIES = 2


async def _prompt_user(prompt: str) -> str:
    """Prompt for user input without blocking the async event loop."""
    return await asyncio.to_thread(input, prompt)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _agent_result_to_text(result: Any) -> str:
    """
    Convert Agent result into text safely.
    """
    if result is None:
        return ""

    if hasattr(result, "text") and result.text:
        return str(result.text)

    if hasattr(result, "messages"):
        try:
            parts = []

            for msg in result.messages:
                if getattr(msg, "text", None):
                    parts.append(msg.text)

            if parts:
                return "\n".join(parts)

        except Exception:
            pass

    return str(result)


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from agent text.

    Supports:
      - pure JSON
      - text containing JSON
      - fallback error object
    """
    try:
        text = (text or "").strip()

        if not text:
            return {}

        if text.startswith("{"):
            return json.loads(text)

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            return json.loads(match.group())

        return {
            "error": "no_json_found",
            "raw": text,
        }

    except Exception as e:
        return {
            "error": str(e),
            "raw": text,
        }


def _extract_json_from_agent_result(result: Any) -> Dict[str, Any]:
    text = _agent_result_to_text(result)
    return _extract_json_from_text(text)


def _is_paused(payload: Dict[str, Any]) -> bool:
    return isinstance(payload, dict) and payload.get("paused") is True


def _is_error(payload: Dict[str, Any]) -> bool:
    return isinstance(payload, dict) and bool(payload.get("error"))


def _hitl_gate_called(payload: Dict[str, Any]) -> bool:
    """
    Agent output-level audit.

    We do NOT calculate threshold here.
    We only check whether the agent proved it called human_approval_gate_tool.

    Paused output also means HITL gate was called.
    """
    if not isinstance(payload, dict):
        return False

    if payload.get("hitl_gate_called") is True:
        return True

    if payload.get("paused") is True:
        return True

    if isinstance(payload.get("hitl"), dict):
        if payload["hitl"].get("decision") or payload["hitl"].get("approved") is not None:
            return True

    if payload.get("hitl_decision"):
        return True

    return False


def _build_paused_output(payload: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
    """
    Include invoice_id so HITL approval can resolve invoice-scoped records.
    """
    return {
        "status": "paused",
        "workflow_id": workflow_id,
        "stage": payload.get("stage"),
        "token": payload.get("resume_token") or payload.get("token"),
        "invoice_id": payload.get("invoice_id"),
    }


def _build_failed_output(payload: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
    return {
        "status": "failed",
        "workflow_id": workflow_id,
        "stage": payload.get("stage") or "workflow",
        "error": str(payload.get("error") or "Unknown workflow error"),
        "invoice_id": payload.get("invoice_id"),
    }


def _build_pending_hitl_output(pending_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert pending HITL Cosmos record into CLI/API paused output.
    """
    return {
        "status": "paused",
        "workflow_id": pending_item.get("workflow_id"),
        "stage": pending_item.get("stage"),
        "token": pending_item.get("token"),
        "invoice_id": pending_item.get("invoice_id"),
        "context": pending_item.get("context"),
        "source": "pending_hitl",
    }


def _get_next_pending_hitl_output(
    workflow_id: str,
    stage: str = "tab_detection",
) -> Optional[Dict[str, Any]]:
    """
    Get next unresolved invoice-scoped HITL item.

    If WorkflowStore.get_next_pending_hitl does not exist yet,
    this safely returns None.
    """
    store = WorkflowStore()

    if not hasattr(store, "get_next_pending_hitl"):
        logger.warning(
            "[Executor] WorkflowStore.get_next_pending_hitl not implemented. "
            "Skipping pending HITL queue lookup. workflow_id=%s stage=%s",
            workflow_id,
            stage,
        )
        return None

    pending = store.get_next_pending_hitl(
        workflow_id=workflow_id,
        stage=stage,
    )

    if not pending:
        return None

    return _build_pending_hitl_output(pending)


def _get_next_pending_or_terminal(
    workflow_id: str,
    stage: str = "tab_detection",
) -> Optional[Dict[str, Any]]:
    """
    Production helper:
    Returns next pending invoice-level HITL if one exists.
    Otherwise returns None.
    """
    return _get_next_pending_hitl_output(
        workflow_id=workflow_id,
        stage=stage,
    )


def _extract_splits_from_classification(classification: dict) -> list:
    """
    Extract split segment blob names from classification final JSON.

    Expected output shape:
      classification["splitting"]["segment_blob_names"]

    Also supports:
      - splitted_blob_names
      - segment_blob_name as a single string
    """
    if not isinstance(classification, dict):
        return []

    splitting = classification.get("splitting", {}) or {}

    splits = (
        splitting.get("segment_blob_names")
        or splitting.get("splitted_blob_names")
        or []
    )

    if isinstance(splits, str):
        return [splits]

    if isinstance(splits, list) and splits:
        return splits

    single_segment = splitting.get("segment_blob_name")

    if isinstance(single_segment, str) and single_segment:
        return [single_segment]

    return []


def _classification_has_split_output(classification: dict) -> bool:
    """
    Returns True only if classification output contains usable split blob names.
    """
    return bool(_extract_splits_from_classification(classification))


def _invoice_id_from_blob(blob_name: str, classification: dict) -> Optional[str]:
    """
    Resolve invoice_id from classification.splitting using blob name.
    """
    if not isinstance(classification, dict):
        return None

    splitting = classification.get("splitting", {}) or {}

    invoice_ids = splitting.get("invoice_ids", []) or []
    segment_blob_names = splitting.get("segment_blob_names", []) or []

    blob_name_norm = str(blob_name or "").strip()

    for inv, blob in zip(invoice_ids, segment_blob_names):
        blob_norm = str(blob or "").strip()

        if blob_norm == blob_name_norm:
            return str(inv).strip().upper()

        if blob_norm.removesuffix(".json") == blob_name_norm.removesuffix(".json"):
            return str(inv).strip().upper()

    return None


def _blob_from_invoice_id(invoice_id: str, classification: dict) -> Optional[str]:
    """
    Resolve segment blob name from invoice_id using classification.splitting.
    """
    if not isinstance(classification, dict):
        return None

    splitting = classification.get("splitting", {}) or {}

    invoice_ids = splitting.get("invoice_ids", []) or []
    segment_blob_names = splitting.get("segment_blob_names", []) or []

    invoice_id_norm = str(invoice_id or "").strip().upper()

    for inv, blob in zip(invoice_ids, segment_blob_names):
        if str(inv or "").strip().upper() == invoice_id_norm:
            return str(blob or "").strip()

    return None


def _classification_has_required_steps(parsed: Dict[str, Any]) -> bool:
    """
    Output-level audit for classification.
    Checks whether the agent result proves DI extraction, vendor resolution,
    classification, HITL gate, and split completed.

    This does NOT execute any tools.
    """
    if not isinstance(parsed, dict):
        return False

    if _is_paused(parsed) or _is_error(parsed):
        return True

    di_ok = bool(
        parsed.get("di_extraction_completed") is True
        or parsed.get("di_extraction")
        or parsed.get("di_output")
        or parsed.get("di_markdown")
        or parsed.get("markdown")
        or parsed.get("markdown_pages")
    )

    vendor = parsed.get("vendor", {}) or {}

    vendor_ok = bool(
        parsed.get("vendor_resolution_completed") is True
        or vendor.get("selected_vendor")
        or parsed.get("selected_vendor")
    )

    classification_ok = bool(
        parsed.get("classification_completed") is True
        or parsed.get("classification")
    )

    hitl_ok = _hitl_gate_called(parsed)

    split_ok = bool(
        parsed.get("split_invoice_completed") is True
        or (parsed.get("splitting", {}) or {}).get("segment_blob_names")
    )

    return di_ok and vendor_ok and classification_ok and hitl_ok and split_ok


def _get_classification_context(workflow_id: str, fallback_market: str = "") -> Dict[str, Any]:
    """
    Load classification_final and derive customer_name, invoice_type, market.
    Used for retrying one or more failed invoices.
    """
    store = WorkflowStore()

    classification = store.load_stage_output(
        workflow_id,
        "classification_final",
    ) or {}

    if not classification:
        return {
            "classification": {},
            "customer_name": None,
            "invoice_type": None,
            "market_name": fallback_market,
        }

    customer_name = classification.get("vendor", {}).get("selected_vendor")
    invoice_type = classification.get("classification", {}).get("Type_of_Invoice")
    market_name = classification.get("market") or fallback_market

    if not market_name:
        meta = store.load_run_meta(workflow_id) or {}
        market_name = meta.get("market")

    return {
        "classification": classification,
        "customer_name": customer_name,
        "invoice_type": invoice_type,
        "market_name": market_name,
    }


def _build_classification_correction_prompt(
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    previous_output: Dict[str, Any],
) -> str:
    """
    Agentic correction prompt.
    Python does not call tools directly. It asks the classification agent
    to correct skipped mandatory steps.
    """
    return f"""
Your previous classification response did not prove that all required classification steps were completed.

This is an agentic correction request. You must use tools yourself.

Inputs:
- workflow_id: {workflow_id}
- pdf_blob_name: {pdf_blob_name}
- market: {market}

Previous output:
{json.dumps(previous_output, default=str, indent=2)}

Required classification flow:
1. run_di_extraction
2. resolve_vendor
3. get_classification_rules
4. classify_invoice
5. human_approval_gate_tool
6. split_invoice only if human_approval_gate_tool returns approved=true

Important HITL rule:
- Do NOT calculate threshold yourself.
- Do NOT decide auto-approval yourself.
- human_approval_gate_tool is the only decision gate.
- You must call human_approval_gate_tool after classify_invoice.
- Pass Confidence from classify_invoice as confidence_score.
- If human_approval_gate_tool returns paused=true, stop and return paused JSON.
- If human_approval_gate_tool returns approved=false and paused=false, return rejected JSON.
- If human_approval_gate_tool returns approved=true, continue to split_invoice.

Rules:
- Do not skip run_di_extraction.
- Do not skip resolve_vendor.
- Do not call get_classification_rules before resolve_vendor.
- Do not call classify_invoice before run_di_extraction.
- Do not skip human_approval_gate_tool.
- If approved=true from human_approval_gate_tool, split_invoice is mandatory.
- Return valid JSON only.

Final approved JSON must include:
- di_extraction_completed: true
- vendor_resolution_completed: true
- classification_completed: true
- split_invoice_completed: true
- hitl_gate_called: true
- hitl_decision
- hitl_reason
- vendor.selected_vendor
- classification.Type_of_Invoice
- classification.Confidence
- splitting.segment_blob_names
- splitting.invoice_ids
"""


def _build_indexing_hitl_correction_prompt(
    workflow_id: str,
    blob_name: str,
    invoice_id: str,
    market_name: str,
    customer_name: str,
    invoice_type: str,
    previous_output: Dict[str, Any],
) -> str:
    """
    Agentic HITL correction prompt.
    Python does not call HITL directly. It asks the indexing agent to correct
    the skipped HITL gate step.
    """
    return f"""
Your previous indexing response did not prove that human_approval_gate_tool was called.

This is an agentic correction request. You must use tools yourself.

For this invoice:
- workflow_id: {workflow_id}
- invoice_id: {invoice_id}
- segment_blob_name: {blob_name}
- blob_name: {blob_name}
- market_name: {market_name}
- customer_name: {customer_name}
- invoice_type: {invoice_type}

Previous output:
{json.dumps(previous_output, default=str, indent=2)}

Important HITL rule:
- Do NOT calculate threshold yourself.
- Do NOT decide auto-approval yourself.
- human_approval_gate_tool is the only decision gate.
- You must call human_approval_gate_tool after detect_invoice_tab.
- Pass confidence_score from detect_invoice_tab.
- Pass stage = "tab_detection".
- Pass workflow_id = "{workflow_id}".
- Pass invoice_id = "{invoice_id}".
- If human_approval_gate_tool returns paused=true, stop and return the HITL output.
- If human_approval_gate_tool returns approved=false and paused=false, return rejected JSON.
- If human_approval_gate_tool returns approved=true, continue to load_extraction_rules, extract_invoice_fields, and save_invoice_output.

Required correction:
1. Use the existing invoice context if available.
2. If needed, call fetch_di_markdown and detect_invoice_tab again.
3. Call human_approval_gate_tool.
4. Follow the exact output of human_approval_gate_tool.
5. Do not create duplicate HITL records manually.
6. Do not call extraction tools before tab_detection approval is established.
7. Return valid JSON only.

Final completed JSON must include:
- invoice_id
- tab_name
- confidence_score
- hitl_gate_called: true
- hitl_decision
- hitl_reason
- json_blob_url
"""


def _build_resume_context_message(
    store: WorkflowStore,
    workflow_id: str,
    invoice_id: str = "",
) -> str:
    """
    Build resume context from workflow_state.
    """
    state = store.load_workflow_state(workflow_id)

    if not state:
        return ""

    invoice_id_norm = str(invoice_id or "").strip().upper()

    safe_state = {
        "status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "next_step": state.get("next_step"),
        "invoice_id": invoice_id_norm or state.get("invoice_id"),
        "segment_blob_name": state.get("segment_blob_name"),
        "excel_blob_name": state.get("excel_blob_name"),
        "customer_name": state.get("customer_name"),
        "invoice_type": state.get("invoice_type"),
        "detected_tab": state.get("detected_tab"),
        "user_updates": state.get("user_updates", {}),
        "stage_overrides": state.get("stage_overrides", {}),
        "effective_values": state.get("effective_values", {}),
        "hitl": state.get("hitl", {}),
    }

    invoice_scoped_hint = ""

    if invoice_id_norm:
        invoice_scoped_hint = (
            f"For this resume, the active invoice_id is {invoice_id_norm}. "
            f"When looking for human overrides, first check invoice-scoped keys such as "
            f"'tab_detection:{invoice_id_norm}', then fall back to 'tab_detection'. "
        )

    return (
        "\nResume context from workflow_state:\n"
        f"{json.dumps(safe_state, indent=2)}\n"
        "When resuming any stage, ALWAYS prefer effective_values first, "
        "then stage_overrides, then user_updates, then original stage outputs or tool values. "
        f"{invoice_scoped_hint}"
        "Human-provided field names may differ from downstream tool argument names; "
        "map them to the expected tool schema before calling the tool. "
        "For tab_detection, if effective_values.tab_detection.selected_tab exists, "
        "or effective_values.tab_detection:<invoice_id>.selected_tab exists, "
        "use it as tab_name when calling load_extraction_rules. "
        "Do not pass selected_tab directly to load_extraction_rules.\n"
    )


def _is_indexing_resume_stage(stage: Optional[str]) -> bool:
    return stage in {
        "tab_detection",
        "load_extraction_rules",
        "extract_invoice_fields",
        "save_invoice_output",
        "indexing",
    }


async def _run_single_indexing_invoice(
    workflow_id: str,
    blob_name: str,
    market_name: str,
    customer_name: str,
    invoice_type: str,
    classification: Dict[str, Any],
    is_resume: bool = False,
    resume_invoice_id: str = "",
    retry: bool = False,
) -> Dict[str, Any]:
    """
    Runs indexing for exactly one invoice blob.

    Used by:
      - normal parallel indexing
      - retry single failed invoice
      - retry multiple failed invoices

    Does not run classification.
    """
    store = WorkflowStore()

    if not blob_name.endswith(".json"):
        blob_name = f"{blob_name}.json"

    invoice_id_for_blob = (
        _invoice_id_from_blob(blob_name, classification)
        or str(resume_invoice_id or "").strip().upper()
    )

    agent = build_indexing_agent()

    base_prompt = f"""
Execute indexing.

Inputs:
- workflow_id: {workflow_id}
- segment_blob_name: {blob_name}
- blob_name: {blob_name}
- market_name: {market_name}
- customer_name: {customer_name}
- invoice_type: {invoice_type}
"""

    if invoice_id_for_blob:
        base_prompt += f"- invoice_id: {invoice_id_for_blob}\n"

    if retry:
        base_prompt += "- retry: true\n"

    if is_resume:
        base_prompt += "- resume: true\n"
        base_prompt += _build_resume_context_message(
            store,
            workflow_id,
            invoice_id_for_blob or resume_invoice_id,
        )

    prompt = base_prompt
    parsed: Dict[str, Any] = {}

    for attempt in range(1, MAX_INDEXING_HITL_AUDIT_RETRIES + 2):
        result = await agent.run(prompt)

        logger.info(
            "Indexing agent completed workflow_id=%s blob_name=%s invoice_id=%s attempt=%s retry=%s",
            workflow_id,
            blob_name,
            invoice_id_for_blob,
            attempt,
            retry,
        )

        parsed = _extract_json_from_agent_result(result)
        parsed["_blob_name"] = blob_name

        if invoice_id_for_blob and not parsed.get("invoice_id"):
            parsed["invoice_id"] = invoice_id_for_blob

        logger.info(
            "Indexing parsed result workflow_id=%s blob=%s invoice_id=%s attempt=%s parsed=%s",
            workflow_id,
            blob_name,
            parsed.get("invoice_id"),
            attempt,
            json.dumps(parsed, default=str),
        )

        if _is_paused(parsed):
            return parsed

        if _is_error(parsed):
            return parsed

        if _hitl_gate_called(parsed):
            return parsed

        if attempt <= MAX_INDEXING_HITL_AUDIT_RETRIES:
            audit_invoice_id = (
                invoice_id_for_blob
                or parsed.get("invoice_id")
                or ""
            )

            logger.warning(
                "Indexing HITL gate audit failed. Agent did not prove "
                "human_approval_gate_tool was called. Retrying agentically. "
                "workflow_id=%s invoice_id=%s attempt=%s",
                workflow_id,
                audit_invoice_id,
                attempt,
            )

            prompt = _build_indexing_hitl_correction_prompt(
                workflow_id=workflow_id,
                blob_name=blob_name,
                invoice_id=audit_invoice_id,
                market_name=market_name,
                customer_name=customer_name,
                invoice_type=invoice_type,
                previous_output=parsed,
            )

            continue

        return {
            "error": (
                "indexing agent skipped human_approval_gate_tool "
                "or did not prove it was called"
            ),
            "stage": "tab_detection_hitl_gate_audit",
            "invoice_id": invoice_id_for_blob or parsed.get("invoice_id"),
            "blob_name": blob_name,
            "details": (
                "The LLM agent must call human_approval_gate_tool after "
                "detect_invoice_tab. Threshold decision belongs to the HITL tool, "
                "not executor and not the LLM."
            ),
            "last_agent_output": parsed,
        }

    return parsed


# -------------------------------------------------------------------
# Executor 1: Classification
# -------------------------------------------------------------------

@executor(id="run_classification")
async def run_classification(
    ctx_input: dict,
    ctx: WorkflowContext[dict],
):
    workflow_id = ctx_input["workflow_id"]
    store = WorkflowStore()

    resume_from_stage = ctx_input.get("resume_from_stage")
    resume_invoice_id = ctx_input.get("invoice_id") or ""

    if ctx_input.get("is_resume") and _is_indexing_resume_stage(resume_from_stage):
        cached_classification = store.load_stage_output(
            workflow_id,
            "classification_final",
        )

        if cached_classification:
            logger.info(
                "Skipping classification on resume. workflow_id=%s "
                "resume_from_stage=%s invoice_id=%s",
                workflow_id,
                resume_from_stage,
                resume_invoice_id,
            )

            ctx_input["classification"] = cached_classification
            await ctx.send_message(ctx_input)
            return

        logger.warning(
            "No classification_final cache found. Classification will run again. "
            "workflow_id=%s resume_from_stage=%s invoice_id=%s",
            workflow_id,
            resume_from_stage,
            resume_invoice_id,
        )

    with tracer.start_as_current_span("Classification_agent") as span:
        span.add_event(
            name="Classification started",
            attributes={
                "workflow_id": workflow_id,
                "input": json.dumps(ctx_input, default=str),
            },
        )

        agent = build_classification_agent()

        try:
            pdf_blob_name = ctx_input["pdf_blob_name"]
            market = ctx_input["market"]
            is_resume = bool(ctx_input.get("is_resume"))

            input_text = f"""
Process invoice classification.

Inputs:
- workflow_id: {workflow_id}
- pdf_blob_name: {pdf_blob_name}
- market: {market}
"""

            if is_resume:
                input_text += "- resume: true\n"
                input_text += _build_resume_context_message(
                    store,
                    workflow_id,
                    resume_invoice_id,
                )

            parsed: Dict[str, Any] = {}

            for attempt in range(1, MAX_CLASSIFICATION_AUDIT_RETRIES + 2):
                result = await agent.run(input_text)

                logger.info(
                    "Classification agent completed workflow_id=%s attempt=%s",
                    workflow_id,
                    attempt,
                )

                parsed = _extract_json_from_agent_result(result)

                if _is_paused(parsed) or _is_error(parsed):
                    break

                if _classification_has_required_steps(parsed):
                    break

                if attempt <= MAX_CLASSIFICATION_AUDIT_RETRIES:
                    logger.warning(
                        "Classification audit failed. Required steps/HITL gate missing. "
                        "Retrying agentically. workflow_id=%s attempt=%s parsed=%s",
                        workflow_id,
                        attempt,
                        json.dumps(parsed, default=str),
                    )

                    input_text = _build_classification_correction_prompt(
                        workflow_id=workflow_id,
                        pdf_blob_name=pdf_blob_name,
                        market=market,
                        previous_output=parsed,
                    )
                    continue

                parsed = {
                    "status": "failed",
                    "error": (
                        "classification agent skipped required steps or HITL gate: "
                        "di_extraction/vendor_resolution/classify_invoice/"
                        "human_approval_gate_tool/split_invoice"
                    ),
                    "stage": "classification_audit",
                    "last_agent_output": parsed,
                }
                break

        except Exception as e:
            parsed = {
                "error": str(e),
                "stage": "classification",
            }

        ctx_input["classification"] = parsed

        active_step = store.load_stage_output(workflow_id, "ACTIVE_STEP")

        if active_step and active_step.get("status") == "paused":
            paused_payload = {
                "paused": True,
                "stage": active_step.get("stage"),
                "resume_token": active_step.get("resume_token"),
                "workflow_id": workflow_id,
                "invoice_id": active_step.get("invoice_id"),
            }

            ctx_input["classification"] = paused_payload
            ctx_input.update(_build_paused_output(paused_payload, workflow_id))

            span.add_event(
                name="Classification paused from ACTIVE_STEP",
                attributes={
                    "workflow_id": workflow_id,
                    "active_step": json.dumps(active_step, default=str),
                },
            )

            await ctx.yield_output(ctx_input)
            return

        if _is_paused(parsed):
            paused_output = _build_paused_output(parsed, workflow_id)
            ctx_input.update(paused_output)

            span.add_event(
                name="Classification paused",
                attributes={
                    "workflow_id": workflow_id,
                    "output": json.dumps(parsed, default=str),
                },
            )

            await ctx.yield_output(ctx_input)
            return

        if _is_error(parsed):
            failed_output = _build_failed_output(parsed, workflow_id)

            store.save_failure(
                workflow_id,
                failed_output["stage"],
                failed_output["error"],
            )

            ctx_input.update(failed_output)

            span.add_event(
                name="Classification failed",
                attributes={
                    "workflow_id": workflow_id,
                    "error": failed_output["error"],
                },
            )

            await ctx.yield_output(ctx_input)
            return

        if not _classification_has_split_output(parsed):
            error_message = (
                "split_invoice was not called or classification output "
                "does not contain splitting.segment_blob_names"
            )

            store.save_failure(
                workflow_id,
                "split_invoice",
                error_message,
            )

            ctx_input["status"] = "failed"
            ctx_input["stage"] = "split_invoice"
            ctx_input["error"] = error_message

            span.add_event(
                name="Classification split guard failed",
                attributes={
                    "workflow_id": workflow_id,
                    "error": error_message,
                    "classification": json.dumps(parsed, default=str),
                },
            )

            await ctx.yield_output(ctx_input)
            return

        store.save_stage_output(
            workflow_id,
            "classification_final",
            parsed,
        )

        span.add_event(
            name="Classification completed",
            attributes={
                "workflow_id": workflow_id,
                "output": json.dumps(parsed, default=str),
            },
        )

        await ctx.send_message(ctx_input)


# -------------------------------------------------------------------
# Executor 2: Parallel Indexing
# -------------------------------------------------------------------

@executor(id="run_parallel_indexing")
async def run_parallel_indexing(
    ctx_input: dict,
    ctx: WorkflowContext[dict],
):
    workflow_id = ctx_input["workflow_id"]
    store = WorkflowStore()

    with tracer.start_as_current_span("indexing_agent_in_parallel") as span:
        classification = ctx_input.get("classification", {}) or {}

        if ctx_input.get("status") in {"paused", "failed", "rejected", "stale_token"}:
            await ctx.yield_output(ctx_input)
            return

        if _is_paused(classification):
            paused_output = _build_paused_output(classification, workflow_id)
            ctx_input.update(paused_output)
            await ctx.yield_output(ctx_input)
            return

        if _is_error(classification):
            failed_output = _build_failed_output(classification, workflow_id)

            store.save_failure(
                workflow_id,
                failed_output["stage"],
                failed_output["error"],
            )

            ctx_input.update(failed_output)
            await ctx.yield_output(ctx_input)
            return

        splits = _extract_splits_from_classification(classification)

        if not splits:
            error_message = "No split invoice blobs found from classification output"

            store.save_failure(
                workflow_id,
                "split_invoice",
                error_message,
            )

            ctx_input.update(
                {
                    "status": "failed",
                    "workflow_id": workflow_id,
                    "stage": "split_invoice",
                    "error": error_message,
                    "indexing_results": [],
                }
            )

            await ctx.yield_output(ctx_input)
            return

        customer_name = classification.get("vendor", {}).get("selected_vendor")
        invoice_type = classification.get("classification", {}).get("Type_of_Invoice")
        market_name = classification.get("market") or ctx_input.get("market")
        is_resume = bool(ctx_input.get("is_resume"))
        resume_invoice_id = str(ctx_input.get("invoice_id") or "").strip().upper()

        if is_resume and resume_invoice_id and _is_indexing_resume_stage(
            ctx_input.get("resume_from_stage")
        ):
            selected_blob = _blob_from_invoice_id(resume_invoice_id, classification)

            if selected_blob:
                splits = [selected_blob]

                logger.info(
                    "Resume scoped to invoice_id=%s blob=%s workflow_id=%s",
                    resume_invoice_id,
                    selected_blob,
                    workflow_id,
                )

                span.add_event(
                    name="Indexing resume scoped to invoice",
                    attributes={
                        "workflow_id": workflow_id,
                        "invoice_id": resume_invoice_id,
                        "blob": selected_blob,
                    },
                )
            else:
                error_message = (
                    f"Resume invoice_id={resume_invoice_id} not found in "
                    "classification split output"
                )

                store.save_failure(
                    workflow_id,
                    "resume_invoice_filter",
                    error_message,
                )

                ctx_input.update(
                    {
                        "status": "failed",
                        "workflow_id": workflow_id,
                        "stage": "resume_invoice_filter",
                        "error": error_message,
                        "invoice_id": resume_invoice_id,
                        "indexing_results": [],
                    }
                )

                await ctx.yield_output(ctx_input)
                return

        span.add_event(
            name="Indexing started",
            attributes={
                "workflow_id": workflow_id,
                "input": json.dumps(classification, default=str),
                "is_resume": is_resume,
                "resume_invoice_id": resume_invoice_id,
                "splits": json.dumps(splits, default=str),
            },
        )

        results = await asyncio.gather(
            *[
                _run_single_indexing_invoice(
                    workflow_id=workflow_id,
                    blob_name=s,
                    market_name=market_name,
                    customer_name=customer_name,
                    invoice_type=invoice_type,
                    classification=classification,
                    is_resume=is_resume,
                    resume_invoice_id=resume_invoice_id,
                    retry=False,
                )
                for s in splits
            ],
            return_exceptions=True,
        )

        successful_results = []
        failed = []

        for idx, result in enumerate(results):
            source_blob = splits[idx]
            source_invoice_id = _invoice_id_from_blob(source_blob, classification)

            if isinstance(result, Exception):
                failed.append(
                    {
                        "blob_name": source_blob,
                        "stage": "indexing",
                        "error": str(result),
                        "invoice_id": source_invoice_id,
                    }
                )
                continue

            if source_invoice_id and not result.get("invoice_id"):
                result["invoice_id"] = source_invoice_id

            if _is_paused(result):
                paused_output = _build_paused_output(result, workflow_id)

                ctx_input.update(paused_output)
                ctx_input["indexing_results"] = successful_results
                ctx_input["partial_results"] = successful_results
                ctx_input["paused_blob"] = source_blob

                span.add_event(
                    name="Indexing paused",
                    attributes={
                        "workflow_id": workflow_id,
                        "output": json.dumps(result, default=str),
                    },
                )

                await ctx.yield_output(ctx_input)
                return

            if _is_error(result):
                failed.append(
                    {
                        "blob_name": source_blob,
                        "stage": result.get("stage") or "indexing",
                        "error": str(result.get("error")),
                        "invoice_id": (
                            result.get("invoice_id")
                            or source_invoice_id
                        ),
                        "raw": result.get("raw"),
                        "details": result.get("details"),
                    }
                )
                continue

            successful_results.append(result)

        ctx_input["indexing_results"] = successful_results

        if failed:
            first_error = failed[0]
            stage = first_error.get("stage") or "indexing"
            error_message = first_error.get("error") or "Indexing failed"

            store.save_failure(workflow_id, stage, error_message)

            status = "partial_failed" if successful_results else "failed"

            ctx_input.update(
                {
                    "status": status,
                    "workflow_id": workflow_id,
                    "stage": stage,
                    "error": error_message,
                    "failed_blobs": failed,
                    "indexing_results": successful_results,
                    "invoice_id": first_error.get("invoice_id"),
                }
            )

            span.add_event(
                name="Indexing partial failed" if successful_results else "Indexing failed",
                attributes={
                    "workflow_id": workflow_id,
                    "status": status,
                    "error": error_message,
                    "failed_blobs": json.dumps(failed, default=str),
                    "successful_count": len(successful_results),
                    "failed_count": len(failed),
                },
            )

            await ctx.yield_output(ctx_input)
            return

        ctx_input["status"] = "completed"

        span.add_event(
            name="Indexing completed",
            attributes={
                "workflow_id": workflow_id,
                "output": json.dumps(successful_results, default=str),
            },
        )

        await ctx.yield_output(ctx_input)


# -------------------------------------------------------------------
# Build Workflow
# -------------------------------------------------------------------

def create_workflow():
    return (
        WorkflowBuilder(start_executor=run_classification)
        .add_edge(run_classification, run_parallel_indexing)
        .build()
    )


# -------------------------------------------------------------------
# Main executor workflow runner
# -------------------------------------------------------------------

async def run_workflow_executor(
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    is_resume: bool = False,
    resume_from_stage: Optional[str] = None,
    invoice_id: str = "",
) -> Dict[str, Any]:
    store = WorkflowStore()

    meta = store.load_run_meta(workflow_id)

    if not meta:
        store.save_run_meta(workflow_id, pdf_blob_name, market)

    store.update_workflow_state(
        workflow_id,
        {
            "status": "running",
            "invoice_id": invoice_id or None,
        },
    )

    checkpoint_storage = FileCheckpointStorage(workflow_id)

    input_data = {
        "workflow_id": workflow_id,
        "pdf_blob_name": pdf_blob_name,
        "market": market,
        "is_resume": is_resume,
        "resume_from_stage": resume_from_stage,
        "invoice_id": invoice_id or "",
    }

    workflow = create_workflow()

    with tracer.start_as_current_span("workflow_run") as span:
        span.add_event(
            name="Workflow started",
            attributes={
                "workflow_id": workflow_id,
                "pdf_blob_name": pdf_blob_name,
                "market": market,
                "is_resume": is_resume,
                "resume_from_stage": resume_from_stage or "",
                "invoice_id": invoice_id or "",
            },
        )

        events = await workflow.run(
            input_data,
            checkpoint_storage=checkpoint_storage,
        )

        outputs = events.get_outputs()
        final_state = events.get_final_state()

        logger.info("Workflow outputs workflow_id=%s outputs=%s", workflow_id, outputs)
        logger.info("Workflow final_state workflow_id=%s state=%s", workflow_id, final_state)

        span.add_event(
            name="Workflow completed",
            attributes={
                "workflow_id": workflow_id,
                "outputs": json.dumps(outputs, default=str),
                "final_state": json.dumps(final_state, default=str),
            },
        )

        final_output = outputs[-1] if outputs else {}

        if isinstance(final_output, dict):
            if final_output.get("status") == "paused":
                return {
                    "status": "paused",
                    "workflow_id": workflow_id,
                    "stage": final_output.get("stage"),
                    "token": final_output.get("token"),
                    "invoice_id": final_output.get("invoice_id"),
                    "output": final_output,
                }

            if final_output.get("status") == "partial_failed":
                return {
                    "status": "partial_failed",
                    "workflow_id": workflow_id,
                    "stage": final_output.get("stage"),
                    "error": final_output.get("error"),
                    "invoice_id": final_output.get("invoice_id"),
                    "failed_blobs": final_output.get("failed_blobs", []),
                    "indexing_results": final_output.get("indexing_results", []),
                    "output": final_output,
                }

            if final_output.get("status") == "failed":
                return {
                    "status": "failed",
                    "workflow_id": workflow_id,
                    "stage": final_output.get("stage"),
                    "error": final_output.get("error"),
                    "invoice_id": final_output.get("invoice_id"),
                    "failed_blobs": final_output.get("failed_blobs", []),
                    "indexing_results": final_output.get("indexing_results", []),
                    "output": final_output,
                }

            if final_output.get("status") == "stale_token":
                return {
                    "status": "stale_token",
                    "workflow_id": workflow_id,
                    "stage": final_output.get("stage"),
                    "error": final_output.get("error"),
                    "invoice_id": final_output.get("invoice_id"),
                    "output": final_output,
                }

            if final_output.get("status") == "completed":
                store.clear_failure(workflow_id)
                store.mark_completed(workflow_id)

                return {
                    "status": "completed",
                    "workflow_id": workflow_id,
                    "output": final_output,
                }

        active_after = store.load_stage_output(workflow_id, "ACTIVE_STEP")

        if active_after and active_after.get("status") == "paused":
            return {
                "status": "paused",
                "workflow_id": workflow_id,
                "stage": active_after.get("stage"),
                "token": active_after.get("resume_token"),
                "invoice_id": active_after.get("invoice_id"),
            }

        failure = store.load_failure(workflow_id)

        if failure:
            return {
                "status": "failed",
                "workflow_id": workflow_id,
                "stage": failure.get("stage"),
                "error": failure.get("error"),
                "invoice_id": invoice_id or failure.get("invoice_id"),
            }

        store.clear_failure(workflow_id)
        store.mark_completed(workflow_id)

        return {
            "status": "completed",
            "workflow_id": workflow_id,
            "outputs": outputs,
        }


# -------------------------------------------------------------------
# Retry failed invoice(s)
# -------------------------------------------------------------------

async def retry_failed_invoice_executor(
    workflow_id: str,
    blob_name: str,
) -> Dict[str, Any]:
    """
    Retry indexing for only one failed split invoice.

    This does NOT rerun classification.
    This does NOT rerun successful invoices.
    It reuses classification_final to get market/customer/invoice_type.
    """
    store = WorkflowStore()

    context = _get_classification_context(workflow_id)
    classification = context.get("classification") or {}

    if not classification:
        return {
            "status": "failed",
            "workflow_id": workflow_id,
            "stage": "classification_final",
            "error": "classification_final cache not found. Cannot retry single invoice.",
            "blob_name": blob_name,
        }

    market_name = context.get("market_name")
    customer_name = context.get("customer_name")
    invoice_type = context.get("invoice_type")

    if not blob_name.endswith(".json"):
        blob_name = f"{blob_name}.json"

    parsed = await _run_single_indexing_invoice(
        workflow_id=workflow_id,
        blob_name=blob_name,
        market_name=market_name,
        customer_name=customer_name,
        invoice_type=invoice_type,
        classification=classification,
        is_resume=False,
        resume_invoice_id="",
        retry=True,
    )

    parsed["_blob_name"] = blob_name

    if _is_paused(parsed):
        paused_output = _build_paused_output(parsed, workflow_id)
        paused_output["blob_name"] = blob_name
        paused_output["output"] = parsed
        return paused_output

    if _is_error(parsed):
        failed_output = _build_failed_output(parsed, workflow_id)
        failed_output["blob_name"] = blob_name
        failed_output["invoice_id"] = parsed.get("invoice_id")
        failed_output["output"] = parsed

        store.save_failure(
            workflow_id,
            failed_output["stage"],
            failed_output["error"],
        )

        return failed_output

    store.clear_failure(workflow_id)

    store.save_stage_output(
        workflow_id,
        f"retry_result:{blob_name}",
        parsed,
    )

    return {
        "status": "completed",
        "workflow_id": workflow_id,
        "blob_name": blob_name,
        "invoice_id": parsed.get("invoice_id"),
        "output": parsed,
    }


async def retry_failed_invoices_executor(
    workflow_id: str,
    failed_blobs: list,
) -> Dict[str, Any]:
    """
    Retry multiple failed split invoices.

    failed_blobs can be:
      - list of dicts with blob_name
      - list of blob_name strings

    Does NOT rerun classification.
    Does NOT rerun successful invoices.
    """
    normalized_failed_blobs = []

    for item in failed_blobs or []:
        if isinstance(item, str):
            normalized_failed_blobs.append({"blob_name": item})
        elif isinstance(item, dict):
            normalized_failed_blobs.append(item)

    if not normalized_failed_blobs:
        return {
            "status": "failed",
            "workflow_id": workflow_id,
            "stage": "retry",
            "error": "No failed blobs provided for retry.",
            "failed_blobs": [],
        }

    retry_results = await asyncio.gather(
        *[
            retry_failed_invoice_executor(
                workflow_id=workflow_id,
                blob_name=item.get("blob_name"),
            )
            for item in normalized_failed_blobs
            if item.get("blob_name")
        ],
        return_exceptions=True,
    )

    successful_retries = []
    failed_retries = []
    paused_retries = []

    for idx, result in enumerate(retry_results):
        source = normalized_failed_blobs[idx]

        if isinstance(result, Exception):
            failed_retries.append(
                {
                    "blob_name": source.get("blob_name"),
                    "stage": "retry",
                    "error": str(result),
                    "invoice_id": source.get("invoice_id"),
                }
            )
            continue

        if result.get("status") == "paused":
            paused_retries.append(result)
            continue

        if result.get("status") == "failed":
            failed_retries.append(result)
            continue

        if result.get("status") == "completed":
            successful_retries.append(result)
            continue

        failed_retries.append(
            {
                "blob_name": source.get("blob_name"),
                "stage": "retry",
                "error": f"Unexpected retry status: {result.get('status')}",
                "output": result,
            }
        )

    if paused_retries:
        return {
            "status": "paused",
            "workflow_id": workflow_id,
            "pending_pauses": paused_retries,
            "successful_retries": successful_retries,
            "failed_blobs": failed_retries,
        }

    if failed_retries:
        return {
            "status": "partial_failed" if successful_retries else "failed",
            "workflow_id": workflow_id,
            "successful_retries": successful_retries,
            "failed_blobs": failed_retries,
        }

    return {
        "status": "completed",
        "workflow_id": workflow_id,
        "successful_retries": successful_retries,
    }


# -------------------------------------------------------------------
# Resume after HITL approval/rejection
# -------------------------------------------------------------------

async def resume_after_approval_executor(
    workflow_id: str,
    stage: str,
    token: str,
    approved: bool,
    user_payload: Optional[Dict[str, Any]] = None,
    invoice_id: str = "",
) -> Dict[str, Any]:
    store = WorkflowStore()

    meta = store.load_run_meta(workflow_id)

    if not meta:
        return {
            "status": "failed",
            "workflow_id": workflow_id,
            "stage": stage,
            "error": "Run metadata not found for workflow_id",
            "invoice_id": invoice_id or None,
        }

    pdf_blob_name = meta.get("pdf_blob_name")
    market = meta.get("market")

    if not pdf_blob_name or not market:
        return {
            "status": "failed",
            "workflow_id": workflow_id,
            "stage": stage,
            "error": "Missing pdf_blob_name or market in run metadata",
            "invoice_id": invoice_id or None,
        }

    stage_key = (stage or "").strip().lower()
    normalized_invoice_id = (invoice_id or "").strip().upper()

    try:
        await resolve_hitl_decision(
            workflow_id=workflow_id,
            stage=stage_key,
            token=token,
            approved=approved,
            user_payload=user_payload,
            invoice_id=normalized_invoice_id,
        )

    except ValueError as e:
        if "Invalid token" in str(e):
            active_step = store.load_stage_output(workflow_id, "ACTIVE_STEP")

            return {
                "status": "stale_token",
                "workflow_id": workflow_id,
                "stage": stage_key,
                "invoice_id": normalized_invoice_id or None,
                "error": str(e),
                "active_step": active_step,
                "message": (
                    "The HITL token is stale. Refresh workflow status and use the latest token."
                ),
            }

        raise

    if not approved:
        if stage_key == "tab_detection":
            next_pending = _get_next_pending_or_terminal(
                workflow_id=workflow_id,
                stage="tab_detection",
            )

            if next_pending:
                return {
                    **next_pending,
                    "previous_decision": {
                        "status": "rejected",
                        "workflow_id": workflow_id,
                        "stage": stage_key,
                        "invoice_id": normalized_invoice_id or None,
                    },
                }

            return {
                "status": "completed",
                "workflow_id": workflow_id,
                "stage": stage_key,
                "invoice_id": normalized_invoice_id or None,
                "message": "Invoice rejected. No more pending HITL invoices.",
                "previous_decision": {
                    "status": "rejected",
                    "workflow_id": workflow_id,
                    "stage": stage_key,
                    "invoice_id": normalized_invoice_id or None,
                },
            }

        return {
            "status": "rejected",
            "workflow_id": workflow_id,
            "stage": stage_key,
            "invoice_id": normalized_invoice_id or None,
            "message": "Workflow-level HITL was rejected.",
        }

    resumed_result = await run_workflow_executor(
        workflow_id=workflow_id,
        pdf_blob_name=pdf_blob_name,
        market=market,
        is_resume=True,
        resume_from_stage=stage_key,
        invoice_id=normalized_invoice_id,
    )

    if stage_key == "tab_detection" and resumed_result.get("status") == "completed":
        try:
            if hasattr(store, "finalize_invoice_hitl_if_complete"):
                finalize_result = store.finalize_invoice_hitl_if_complete(workflow_id)

                logger.info(
                    "[Executor] finalize check after approved resume workflow=%s result=%s",
                    workflow_id,
                    finalize_result,
                )

                final_state = store.load_workflow_state(workflow_id) or {}

                if finalize_result.get("finalized"):
                    resumed_result["status"] = final_state.get(
                        "status",
                        resumed_result.get("status"),
                    )
                    resumed_result["approved_invoices"] = final_state.get(
                        "approved_invoices",
                        [],
                    )
                    resumed_result["rejected_invoices"] = final_state.get(
                        "rejected_invoices",
                        [],
                    )
                    resumed_result["pending_invoices"] = final_state.get(
                        "pending_invoices",
                        [],
                    )
                    resumed_result["resolved_invoice_count"] = final_state.get(
                        "resolved_invoice_count",
                    )
                    resumed_result["total_invoice_count"] = final_state.get(
                        "total_invoice_count",
                    )

        except Exception:
            logger.exception(
                "[Executor] Failed finalization check workflow=%s",
                workflow_id,
            )

    if stage_key == "tab_detection" and resumed_result.get("status") in {
        "completed",
        "completed_with_rejections",
    }:
        next_pending = _get_next_pending_or_terminal(
            workflow_id=workflow_id,
            stage="tab_detection",
        )

        if next_pending:
            return {
                **next_pending,
                "previous_decision": {
                    "status": "approved",
                    "workflow_id": workflow_id,
                    "stage": stage_key,
                    "invoice_id": normalized_invoice_id or None,
                },
            }

    return resumed_result


# -------------------------------------------------------------------
# Interactive local test
# -------------------------------------------------------------------

async def main():
    workflow_id = f"wf-{uuid.uuid4()}"

    pdf_blob_name = "Albertsons companies-Sample.pdf"
    market = "USR"

    result = await run_workflow_executor(
        workflow_id=workflow_id,
        pdf_blob_name=pdf_blob_name,
        market=market,
        is_resume=False,
    )

    print(result)

    while result["status"] in {
        "paused",
        "failed",
        "partial_failed",
        "stale_token",
        "rejected",
    }:
        if result["status"] == "paused":
            print(f"\nPaused at stage: {result['stage']}")
            print(f"Invoice ID: {result.get('invoice_id')}")
            print(f"Resume token: {result.get('token')}")

            if result.get("context"):
                print("\nContext:")
                print(result.get("context"))

            decision = (await _prompt_user("Press [a] to approve / [r] to reject: ")).strip().lower()
            approved = decision != "r"

            user_payload = {}

            if approved and result["stage"] == "tab_detection":
                selected_tab = (await _prompt_user("Optional: enter selected tab name: ")).strip()

                if selected_tab:
                    user_payload["selected_tab"] = selected_tab

            if approved and result["stage"] == "classification":
                invoice_type = (await _prompt_user("Optional: enter corrected invoice type: ")).strip()

                if invoice_type:
                    user_payload["invoice_type"] = invoice_type

            if approved and result["stage"] == "vendor_resolution":
                selected_vendor = (await _prompt_user("Optional: enter corrected vendor name: ")).strip()

                if selected_vendor:
                    user_payload["selected_vendor"] = selected_vendor

            result = await resume_after_approval_executor(
                workflow_id=result["workflow_id"],
                stage=result["stage"],
                token=result["token"],
                approved=approved,
                user_payload=user_payload or None,
                invoice_id=result.get("invoice_id") or "",
            )

            if result.get("status") in {"completed", "rejected"}:
                next_pending = _get_next_pending_hitl_output(
                    workflow_id=result["workflow_id"],
                    stage="tab_detection",
                )

                if next_pending:
                    print("\nFound next pending HITL invoice.")
                    result = next_pending

        elif result["status"] == "partial_failed":
            print("\nPARTIAL FAILURE")
            print(f"Workflow ID: {result.get('workflow_id')}")
            print(f"Stage: {result.get('stage')}")
            print(f"Error: {result.get('error')}")

            failed_blobs = result.get("failed_blobs", []) or []

            if not failed_blobs:
                print("No failed blobs found.")
                return

            print("\nFailed blobs:")
            for idx, item in enumerate(failed_blobs, start=1):
                print(
                    f"{idx}. blob={item.get('blob_name')} "
                    f"invoice_id={item.get('invoice_id')} "
                    f"stage={item.get('stage')} "
                    f"error={item.get('error')}"
                )

            choice = (await _prompt_user(
                "Enter failed blob number to retry, [a] retry all, [q] quit: "
            )).strip().lower()

            if choice == "q":
                print("Stopped by user")
                return

            if choice == "a":
                result = await retry_failed_invoices_executor(
                    workflow_id=result["workflow_id"],
                    failed_blobs=failed_blobs,
                )
            else:
                try:
                    selected = failed_blobs[int(choice) - 1]
                except Exception:
                    print("Invalid choice")
                    return

                retry_blob = selected.get("blob_name")

                result = await retry_failed_invoice_executor(
                    workflow_id=result["workflow_id"],
                    blob_name=retry_blob,
                )

        elif result["status"] == "failed":
            print(f"\nFAILED at stage: {result['stage']}")
            print(f"Invoice ID: {result.get('invoice_id')}")
            print(f"Error: {result['error']}")

            failed_blobs = result.get("failed_blobs", []) or []

            if failed_blobs:
                print("\nFailed blobs available:")
                for idx, item in enumerate(failed_blobs, start=1):
                    print(
                        f"{idx}. blob={item.get('blob_name')} "
                        f"invoice_id={item.get('invoice_id')} "
                        f"stage={item.get('stage')} "
                        f"error={item.get('error')}"
                    )

                decision = (await _prompt_user(
                    "Press [a] retry all failed blobs / [r] retry workflow / [q] quit: "
                )).strip().lower()

                if decision == "a":
                    result = await retry_failed_invoices_executor(
                        workflow_id=result["workflow_id"],
                        failed_blobs=failed_blobs,
                    )
                elif decision == "r":
                    result = await run_workflow_executor(
                        workflow_id=result["workflow_id"],
                        pdf_blob_name=pdf_blob_name,
                        market=market,
                        is_resume=True,
                        resume_from_stage=result.get("stage"),
                        invoice_id=result.get("invoice_id") or "",
                    )
                else:
                    print("Stopped by user")
                    return
            else:
                decision = (await _prompt_user("Press [r] to retry workflow / [q] to quit: ")).strip().lower()

                if decision == "r":
                    result = await run_workflow_executor(
                        workflow_id=result["workflow_id"],
                        pdf_blob_name=pdf_blob_name,
                        market=market,
                        is_resume=True,
                        resume_from_stage=result.get("stage"),
                        invoice_id=result.get("invoice_id") or "",
                    )
                else:
                    print("Stopped by user")
                    return

            if result.get("status") in {"completed", "rejected"}:
                next_pending = _get_next_pending_hitl_output(
                    workflow_id=result["workflow_id"],
                    stage="tab_detection",
                )

                if next_pending:
                    print("\nFound next pending HITL invoice.")
                    result = next_pending

        elif result["status"] == "stale_token":
            print("\nSTALE HITL TOKEN")
            print(f"Invoice ID: {result.get('invoice_id')}")
            print(result.get("error"))

            result = await run_workflow_executor(
                workflow_id=result["workflow_id"],
                pdf_blob_name=pdf_blob_name,
                market=market,
                is_resume=True,
                resume_from_stage=result.get("stage"),
                invoice_id=result.get("invoice_id") or "",
            )

            if result.get("status") in {"completed", "rejected"}:
                next_pending = _get_next_pending_hitl_output(
                    workflow_id=result["workflow_id"],
                    stage="tab_detection",
                )

                if next_pending:
                    print("\nFound next pending HITL invoice.")
                    result = next_pending

        elif result["status"] == "rejected":
            print(f"\nREJECTED at stage: {result.get('stage')}")
            print(f"Invoice ID: {result.get('invoice_id')}")

            next_pending = _get_next_pending_hitl_output(
                workflow_id=result["workflow_id"],
                stage="tab_detection",
            )

            if next_pending:
                print("\nFound next pending HITL invoice.")
                result = next_pending
            else:
                break

        print(result)

    print(f"\nFINAL STATUS: {result['status']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    asyncio.run(main())