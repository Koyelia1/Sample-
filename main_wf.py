from __future__ import annotations

import asyncio
import inspect
import json
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from queue_manager import queue_manager
from storage.workflow import WorkflowStore
from workflows.runner_service import WorkflowRunnerService

try:
    from agent_framework._feature_stage import ExperimentalWarning

    warnings.filterwarnings(
        "ignore",
        message=r".*\[SKILLS\].*SkillResource is experimental.*",
        category=ExperimentalWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*\[HARNESS\].*MemoryStore is experimental.*",
        category=ExperimentalWarning,
    )
except ImportError:
    warnings.filterwarnings(
        "ignore",
        message=r".*\[SKILLS\].*SkillResource is experimental.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*\[HARNESS\].*MemoryStore is experimental.*",
    )

from queue_manager import queue_manager
from storage.workflow import WorkflowStore
from workflows.runner_service import WorkflowRunnerService


def _send_terminated_to_queue_if_needed(
    result: Dict[str, Any],
    *,
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    parent_transaction_id: str = "",
) -> bool:
    """
    Send one attachment-level deterministic termination notification.

    For split-invoice processing, the decision is based on aggregated
    invoice_termination_results. For initial source-file termination, the
    existing final_output["termination"] object is used as a fallback.
    """

    status = str(
        (result or {}).get("status")
        or ""
    ).strip().lower()

    if status != "terminated":
        return False

    final_output = result.get(
        "final_output"
    )

    if not isinstance(
        final_output,
        dict,
    ):
        final_output = {}

        outputs = result.get(
            "outputs"
        )

        if isinstance(outputs, list):
            for output in reversed(outputs):
                if isinstance(output, dict):
                    final_output = output
                    break

    invoice_termination_results = (
        final_output.get(
            "invoice_termination_results"
        )
    )

    if not isinstance(
        invoice_termination_results,
        list,
    ):
        invoice_termination_results = []

    terminated_invoice_results = [
        item
        for item in invoice_termination_results
        if isinstance(item, dict)
        and bool(
            item.get(
                "termination_required",
                False,
            )
        )
        and bool(
            item.get(
                "termination_found",
                False,
            )
        )
        and bool(
            item.get(
                "termination_executed",
                False,
            )
        )
    ]

    termination = final_output.get(
        "termination"
    )

    if not isinstance(
        termination,
        dict,
    ):
        termination = {}

    file_level_termination = bool(
        termination.get(
            "termination_required",
            False,
        )
        and termination.get(
            "termination_found",
            False,
        )
        and (
            termination.get(
                "termination_executed",
                False,
            )
            or final_output.get(
                "termination_executed",
                False,
            )
        )
    )

    if (
        not terminated_invoice_results
        and not file_level_termination
    ):
        return False

    termination_summary = final_output.get(
        "termination_summary"
    )

    if not isinstance(
        termination_summary,
        dict,
    ):
        termination_summary = {}

    terminated_invoice_ids = [
        str(
            item.get(
                "invoice_id",
                "",
            )
            or ""
        ).strip()
        for item in terminated_invoice_results
        if str(
            item.get(
                "invoice_id",
                "",
            )
            or ""
        ).strip()
    ]

    termination_categories = list(
        dict.fromkeys(
            str(
                item.get(
                    "category",
                    "",
                )
                or ""
            ).strip()
            for item in terminated_invoice_results
            if str(
                item.get(
                    "category",
                    "",
                )
                or ""
            ).strip()
        )
    )

    if not termination_categories:
        file_level_category = str(
            termination.get(
                "category"
            )
            or final_output.get(
                "termination_category"
            )
            or ""
        ).strip()

        if file_level_category:
            termination_categories = [
                file_level_category
            ]

    queue_payload = {
        "attachment_id": workflow_id,
        "status": "terminated",
        "file_path": pdf_blob_name,
        "parent_transaction_id": parent_transaction_id,
        "market": market,
    }

    queue_manager.send_message_review_queue(
        json.dumps(
            queue_payload
        )
    )

    print(
        "[TERMINATED_QUEUE_MESSAGE_SENT] "
        f"workflow_id={workflow_id} "
        f"terminated_invoice_count="
        f"{len(terminated_invoice_results)} "
        f"terminated_invoice_ids="
        f"{terminated_invoice_ids} "
        f"categories={termination_categories} "
        f"aggregate_status="
        f"{termination_summary.get('status', status)}"
    )

    return True


def _validate_workflow_result(result: Any) -> Dict[str, Any]:
    """Ensure WorkflowRunnerService returned the expected dictionary."""
    if not isinstance(result, dict):
        raise TypeError(
            "WorkflowRunnerService must return a dictionary. "
            f"Received: {type(result).__name__}"
        )
    return result


def _get_hitl_token_from_request(request: Dict[str, Any]) -> str:
    """
    Extract the Cosmos HITL token from a direct Agent Framework
    request or an indexing HITL proxy request.
    """
    if not isinstance(request, dict):
        return ""

    review_payload = request.get("review_payload") or {}
    if not isinstance(review_payload, dict):
        review_payload = {}

    hitl_token = str(
        review_payload.get("hitl_token")
        or request.get("hitl_token")
        or ""
    ).strip()

    if hitl_token:
        return hitl_token

    child_payload = (
        review_payload.get("child_review_payload")
        or review_payload.get("original_review_payload")
        or review_payload.get("review_payload")
        or {}
    )

    if not isinstance(child_payload, dict):
        return ""

    return str(child_payload.get("hitl_token") or "").strip()


def _attach_framework_references_to_hitl_records(
    store: WorkflowStore,
    workflow_id: str,
    result: Dict[str, Any],
) -> int:
    """
    Attach checkpoint_id and request_id to every pending Cosmos HITL record.

    Multiple requests may share one checkpoint_id, but each request has its
    own request_id and hitl_token.
    """
    checkpoint_id = str(result.get("checkpoint_id") or "").strip()
    pending_requests = result.get("pending_requests") or []

    if not checkpoint_id:
        if pending_requests:
            raise ValueError(
                "Workflow returned pending requests without a checkpoint_id"
            )
        return 0

    if not isinstance(pending_requests, list):
        raise TypeError("result.pending_requests must be a list")

    updated_count = 0
    update_errors: List[Dict[str, str]] = []

    for request in pending_requests:
        if not isinstance(request, dict):
            continue

        request_id = str(request.get("request_id") or "").strip()
        hitl_token = _get_hitl_token_from_request(request)

        if not request_id:
            update_errors.append(
                {
                    "request_id": "",
                    "hitl_token": hitl_token,
                    "error": "request_id is missing",
                }
            )
            continue

        if not hitl_token:
            update_errors.append(
                {
                    "request_id": request_id,
                    "hitl_token": "",
                    "error": "hitl_token is missing",
                }
            )
            continue

        try:
            updated_record = store.attach_framework_reference_to_hitl(
                workflow_id=workflow_id,
                hitl_token=hitl_token,
                checkpoint_id=checkpoint_id,
                request_id=request_id,
            )
            updated_count += 1

            invoice_id = None
            stage = None
            if isinstance(updated_record, dict):
                invoice_id = updated_record.get("invoice_id")
                stage = updated_record.get("stage")

            print(
                "[HITL_FRAMEWORK_REFERENCE_ATTACHED] "
                f"workflow_id={workflow_id} "
                f"invoice_id={invoice_id} "
                f"stage={stage} "
                f"checkpoint_id={checkpoint_id} "
                f"request_id={request_id} "
                f"hitl_token={hitl_token}"
            )
        except Exception as exc:
            update_errors.append(
                {
                    "request_id": request_id,
                    "hitl_token": hitl_token,
                    "error": str(exc),
                }
            )

    if update_errors:
        print(
            "[HITL_FRAMEWORK_REFERENCE_ERRORS] "
            f"workflow_id={workflow_id} errors={update_errors}"
        )

    print(
        "[HITL_FRAMEWORK_REFERENCE_SUMMARY] "
        f"workflow_id={workflow_id} "
        f"pending_count={len(pending_requests)} "
        f"updated_count={updated_count}"
    )

    return updated_count


def _pending_request_is_tool_failure(request: Any) -> bool:
    """
    Return True when a pending request is a RetryableToolExecutor
    tool-failure escalation rather than a genuine business HITL.

    Covers both a direct escalation (task_type == "tool_failure_escalation")
    and an indexing dispatch proxy carrying a failed child
    (child_task_type == "tool_failure_escalation").
    """
    if not isinstance(request, dict):
        return False

    payload = request.get("review_payload")

    if not isinstance(payload, dict):
        payload = request

    return "tool_failure_escalation" in {
        str(payload.get("task_type") or "").strip().lower(),
        str(payload.get("child_task_type") or "").strip().lower(),
    }


def _nested_lookup(value: Any, key: str, depth: int = 6) -> Any:
    """First non-empty value for `key`, searching wrapped request payloads."""
    if depth < 0 or not isinstance(value, dict):
        return None

    direct = value.get(key)
    if direct not in (None, ""):
        return direct

    for nested in (
        "review_payload",
        "request_data",
        "data",
        "child_review_payload",
        "child_request_data",
        "hitl_payload",
        "current_payload",
    ):
        if nested in value:
            found = _nested_lookup(value[nested], key, depth - 1)
            if found not in (None, ""):
                return found

    return None


def _tool_failure_details(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a per-invoice failure list ({invoice_id, stage, failure_category,
    error}) from the workflow result.

    Prefers the dispatch finalizer's failed_invoice_details; falls back to
    the pending tool-failure escalation payloads (direct or proxied).
    """
    result = result or {}

    final_output = result.get("final_output")
    if not isinstance(final_output, dict):
        final_output = {}

    summary = final_output.get("termination_summary")
    if not isinstance(summary, dict):
        summary = {}

    for source in (result, final_output, summary):
        details = source.get("failed_invoice_details")
        if isinstance(details, list) and details:
            return [d for d in details if isinstance(d, dict)]

    details: List[Dict[str, Any]] = []
    for request in result.get("pending_requests") or []:
        if not _pending_request_is_tool_failure(request):
            continue

        payload = request.get("review_payload") if isinstance(request, dict) else None
        if not isinstance(payload, dict):
            payload = request if isinstance(request, dict) else {}

        details.append(
            {
                "invoice_id": _nested_lookup(payload, "invoice_id"),
                "stage": _nested_lookup(payload, "step_name")
                or _nested_lookup(payload, "stage")
                or "indexing",
                "failure_category": _nested_lookup(payload, "failure_category"),
                "error": _nested_lookup(payload, "last_error")
                or _nested_lookup(payload, "error"),
            }
        )

    return details


def _map_result_to_run_meta_status(
    result: Dict[str, Any],
    completed_status: str = "Completed",
) -> str:
    """
    Map the framework result to the user-facing run_meta status.

    completed_status is "Completed" for an initial run and "Audit" for
    successful execution after HITL resume.
    """
    status = str((result or {}).get("status") or "").strip().lower()
    pending_requests = (result or {}).get("pending_requests") or []

    # A run that is paused ONLY on tool-failure escalations is a technical
    # failure, not an operator review. Report it as "Failed" (with the
    # failing invoice/stage detail recorded on run_meta by the dispatch
    # finalizer) instead of "HITL".
    if pending_requests and all(
        _pending_request_is_tool_failure(request)
        for request in pending_requests
    ):
        return "Failed"

    if (
        status in {"paused", "pending_review", "stale_token"}
        or bool(pending_requests)
    ):
        return "HITL"

    if status == "failed":
        return "Failed"

    if status in {"rejected", "terminated"}:
        return "Terminated"


    if status in {"completed", "completed_with_rejections" ,"query_raised"}:
        return completed_status

    # if status == "query_raised":
    #     return "Query Raised"
    # if status == "query_raised":
    #     return "Query Raised"

    return "In-Progress"


def _update_run_meta_from_result(
    store: WorkflowStore,
    workflow_id: str,
    result: Dict[str, Any],
    completed_status: str = "Completed",
) -> str:
    """Update run_meta from the workflow result and return its new status."""
    run_meta_status = _map_result_to_run_meta_status(
        result=result,
        completed_status=completed_status,
    )

    extra_fields: Dict[str, Any] = {}

    if run_meta_status == "Failed":
        failure_details = _tool_failure_details(result)
        if failure_details:
            extra_fields["failed_invoice_details"] = failure_details
            extra_fields["failed_invoice_count"] = len(failure_details)
            extra_fields["failed_invoice_ids"] = [
                detail.get("invoice_id")
                for detail in failure_details
                if detail.get("invoice_id")
            ]

    store.update_run_meta_status(
        workflow_id, run_meta_status, **extra_fields
    )

    print(
        "[RUN_META_STATUS_UPDATED] "
        f"workflow_id={workflow_id} "
        f"workflow_status={result.get('status')} "
        f"run_meta_status={run_meta_status}"
    )

    return run_meta_status


async def _finalize_execution_timings(
    store: WorkflowStore,
    workflow_id: str,
    run_meta_status: str,
) -> None:
    """Stamp workflow_total / completed_at into run_meta once the run is terminal.

    Skipped while the run is still paused (HITL) or in progress -- those states
    are not the end of the run. Idempotent and also invoked by the indexing
    dispatch finalizer, so calling it here on non-indexing terminal paths
    (early termination, rejection, failure) is safe.
    """
    if str(run_meta_status or "").strip() in {
        "HITL",
        "HITL Triggered",
        "In-Progress",
    }:
        return
    try:
        from utility import exec_timing

        await exec_timing.finalize_totals(workflow_id, store=store)
    except Exception:
        print(f"[EXEC_TIMING_FINALIZE_FAILED] workflow_id={workflow_id}")


async def _close_service(service: WorkflowRunnerService) -> None:
    """Close WorkflowRunnerService when it exposes a close method."""
    close_method = getattr(service, "close", None)
    if close_method is None:
        return

    close_result = close_method()
    if inspect.isawaitable(close_result):
        await close_result


def _send_to_review_queue_if_needed(
    result: Dict[str, Any],
    *,
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    parent_transaction_id: str = "",
) -> bool:
    """Send review-queue HITL notification for review or failure outcomes."""
    status = str((result or {}).get("status") or "").strip().lower()
    has_pending_requests = bool((result or {}).get("pending_requests"))

    # Business rule: paused, pending review, stale token, failed,
    # and rejected outcomes must all be sent to the review queue
    # with queue status "hitl". This is independent of the run_meta
    # mapping, where "failed" -> "Failed" and "rejected" -> "Terminated".
    is_hitl = (
        status
        in {
            "paused",
            "pending_review",
            "stale_token",
            "failed",
            "rejected",
        }
        or has_pending_requests
    )
    if not is_hitl:
        return False

    queue_payload = {
        "attachment_id": workflow_id,
        "status": "hitl",
        "file_path": pdf_blob_name,
        "parent_transaction_id": parent_transaction_id,
        "market": market,
    }

    queue_manager.send_message_review_queue(json.dumps(queue_payload))

    print(
        "[HITL_QUEUE_MESSAGE_SENT] "
        f"workflow_id={workflow_id} workflow_status={status}"
    )
    return True


def _send_failed_to_queue_if_needed(
    result: Dict[str, Any],
    *,
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    parent_transaction_id: str = "",
) -> bool:
    """
    Send one attachment-level failure notification to the review queue.

    Fires for a hard technical failure: workflow status "failed", or a run
    paused only on tool-failure escalations. Payload shape is kept identical
    to the other queue messages; only the status differs ("failed").
    """
    status = str((result or {}).get("status") or "").strip().lower()

    pending_requests = (result or {}).get("pending_requests") or []
    only_tool_failures = bool(pending_requests) and all(
        _pending_request_is_tool_failure(request)
        for request in pending_requests
    )

    if status != "failed" and not only_tool_failures:
        return False

    queue_payload = {
        "attachment_id": workflow_id,
        "status": "failed",
        "file_path": pdf_blob_name,
        "parent_transaction_id": parent_transaction_id,
        "market": market,
    }

    queue_manager.send_message_review_queue(json.dumps(queue_payload))

    print(
        "[FAILED_QUEUE_MESSAGE_SENT] "
        f"workflow_id={workflow_id} "
        f"workflow_status={status or 'tool_failure_escalation'}"
    )
    return True


def _print_result(
    workflow_label: str,
    result: Dict[str, Any],
) -> None:
    status = str(
        (result or {}).get("status")
        or ""
    ).strip().lower()

    if status in {
        "completed",
        "completed_with_rejections",
    }:
        print(
            f"[{workflow_label}] "
            "WORKFLOW COMPLETE"
        )

        for output in result.get(
            "outputs",
            [],
        ):
            print(
                json.dumps(
                    output,
                    indent=2,
                    default=str,
                )
                if isinstance(
                    output,
                    dict,
                )
                else output
            )

        return

    if status == "terminated":
        final_output = result.get(
            "final_output"
        )

        if not isinstance(
            final_output,
            dict,
        ):
            final_output = {}

            outputs = result.get(
                "outputs"
            )

            if isinstance(outputs, list):
                for output in reversed(
                    outputs
                ):
                    if isinstance(
                        output,
                        dict,
                    ):
                        final_output = output
                        break

        termination = final_output.get(
            "termination"
        )

        if not isinstance(
            termination,
            dict,
        ):
            termination = {}

        category = str(
            termination.get(
                "category"
            )
            or final_output.get(
                "termination_category"
            )
            or ""
        ).strip()

        reason = str(
            termination.get(
                "reason"
            )
            or final_output.get(
                "termination_reason"
            )
            or ""
        ).strip()

        stage = str(
            termination.get(
                "termination_stage"
            )
            or final_output.get(
                "termination_stage"
            )
            or ""
        ).strip()

        source = str(
            termination.get(
                "termination_source"
            )
            or final_output.get(
                "termination_source"
            )
            or ""
        ).strip()

        confidence = (
            termination.get(
                "confidence"
            )
        )

        if confidence is None:
            confidence = (
                final_output.get(
                    "termination_confidence",
                    0,
                )
            )

        print(
            f"[{workflow_label}] "
            "WORKFLOW TERMINATED"
        )

        invoice_termination_results = (
            final_output.get(
                "invoice_termination_results"
            )
        )

        if isinstance(
            invoice_termination_results,
            list,
        ):
            terminated_invoice_results = [
                item
                for item in invoice_termination_results
                if isinstance(item, dict)
                and bool(
                    item.get(
                        "termination_required",
                        False,
                    )
                )
                and bool(
                    item.get(
                        "termination_found",
                        False,
                    )
                )
                and bool(
                    item.get(
                        "termination_executed",
                        False,
                    )
                )
            ]

            if terminated_invoice_results:
                print(
                    "  Terminated invoices: "
                    f"{len(terminated_invoice_results)}"
                )

                for item in terminated_invoice_results:
                    print(
                        "  "
                        f"invoice_id={item.get('invoice_id', '')} "
                        f"category={item.get('category', '')} "
                        f"stage={item.get('termination_stage', '')}"
                    )

        # print(
        #     f"  Category   : {category}"
        # )

        # print(
        #     f"  Reason     : {reason}"
        # )

        # print(
        #     f"  Stage      : {stage}"
        # )

        # print(
        #     f"  Source     : {source}"
        # )

        # print(
        #     f"  Confidence : {confidence}"
        # )

        return

    if status in {
        "failed",
        "rejected",
    }:
        print(
            f"[{workflow_label}] "
            f"WORKFLOW FAILED status={status}"
        )

        errors = result.get(
            "errors"
        )

        if not isinstance(errors, list):
            errors = []

        if not errors:
            final_output = result.get(
                "final_output"
            )

            if isinstance(
                final_output,
                dict,
            ):
                error = final_output.get(
                    "error"
                )

                if error:
                    errors = [error]

        for error in errors:
            print(error)

        return

    if status == "query_raised":
        print(
            f"[{workflow_label}] "
            "QUERY RAISED"
        )

        return

    pending = (
        result.get(
            "pending_requests"
        )
        or []
    )

    if not isinstance(
        pending,
        list,
    ):
        pending = []

    print(
        f"[{workflow_label}] "
        "WORKFLOW PAUSED -- "
        f"{len(pending)} pending request(s)"
    )

    print(
        "checkpoint_id to resume with: "
        f"{result.get('checkpoint_id')}"
    )

    for request in pending:
        if not isinstance(
            request,
            dict,
        ):
            continue

        payload = (
            request.get(
                "review_payload"
            )
            or {}
        )

        if not isinstance(
            payload,
            dict,
        ):
            payload = {}

        request_id = request.get(
            "request_id",
            "",
        )

        task_type = payload.get(
            "task_type",
            "unknown",
        )

        stage = (
            payload.get("stage")
            or payload.get("step_name")
            or ""
        )

        invoice_id = payload.get(
            "invoice_id"
        )

        print(
            f"  request_id={request_id} "
            f"task_type={task_type} "
            f"stage={stage} "
            f"invoice_id={invoice_id}"
        )

        message = payload.get(
            "review_message"
        )

        if message:
            print(f"{message}")


def _validate_required(value: Optional[str], field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


async def start_workflow(
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    parent_transaction_id: str = "",
) -> Dict[str, Any]:
    service = WorkflowRunnerService()
    store = WorkflowStore()
    workflow_result_received = False

    payload = {
        "workflow_id": workflow_id,
        "pdf_blob_name": pdf_blob_name,
        "market": market,
        "parent_transaction_id": parent_transaction_id,
    }

    print(f"Workflow ID: {workflow_id}")

    existing_meta = store.load_run_meta(workflow_id)
    if not existing_meta:
        store.save_run_meta(
            workflow_id,
            pdf_blob_name,
            market,
            parent_transaction_id=parent_transaction_id,
        )
        print(
            "[RUN_META_CREATED] "
            f"workflow_id={workflow_id} "
            f"pdf_blob_name={pdf_blob_name} "
            f"market={market} "
            f"parent_transaction_id={parent_transaction_id}"
        )
    else:
        print(f"[RUN_META_ALREADY_EXISTS] workflow_id={workflow_id}")

    store.update_run_meta_status(workflow_id, "In-Progress")

    try:
        result = _validate_workflow_result(await service.start(payload))
        workflow_result_received = True

        pending_requests = result.get("pending_requests") or []
        if not isinstance(pending_requests, list):
            pending_requests = []

        attached_count = _attach_framework_references_to_hitl_records(
            store=store,
            workflow_id=workflow_id,
            result=result,
        )

        _run_meta_status = _update_run_meta_from_result(
            store=store,
            workflow_id=workflow_id,
            result=result,
        )
        await _finalize_execution_timings(
            store, workflow_id, _run_meta_status
        )

        if _send_failed_to_queue_if_needed(
            result=result,
            workflow_id=workflow_id,
            pdf_blob_name=pdf_blob_name,
            market=market,
            parent_transaction_id=parent_transaction_id,
        ):
            pass
        elif pending_requests and attached_count != len(pending_requests):
            print(
                "[REVIEW_QUEUE_MESSAGE_NOT_SENT] "
                f"workflow_id={workflow_id} "
                f"pending_count={len(pending_requests)} "
                f"attached_count={attached_count} "
                "reason=not_all_hitl_records_enriched"
            )
        # else:
            # _send_to_review_queue_if_needed(
            #     result=result,
            #     workflow_id=workflow_id,
            #     pdf_blob_name=pdf_blob_name,
            #     market=market,
            #     parent_transaction_id=parent_transaction_id,
            # )
        else:
            termination_message_sent = (
                _send_terminated_to_queue_if_needed(
                    result=result,
                    workflow_id=workflow_id,
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )
            )

            if not termination_message_sent:
                _send_to_review_queue_if_needed(
                    result=result,
                    workflow_id=workflow_id,
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )

        _print_result(workflow_id, result)
        return result

    except Exception as exc:
        if not workflow_result_received:
            failure_result = {
                "status": "failed",
                "workflow_id": workflow_id,
                "error": str(exc),
            }
            _run_meta_status = _update_run_meta_from_result(
                store=store,
                workflow_id=workflow_id,
                result=failure_result,
            )
            await _finalize_execution_timings(
                store, workflow_id, _run_meta_status
            )
        else:
            print(
                "[POST_WORKFLOW_PROCESSING_FAILED] "
                f"workflow_id={workflow_id} error={exc}"
            )
        raise
    finally:
        await _close_service(service)


async def resume_workflow(
    checkpoint_id: str,
    request_id: str,
    approved: bool,
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    parent_transaction_id: str = "",
    corrections: Optional[Dict[str, Any]] = None,
    comments: Optional[str] = None,
) -> Dict[str, Any]:
    service = WorkflowRunnerService()
    store = WorkflowStore()
    workflow_result_received = False

    review_response = {
        "approved": approved,
        "reviewer": "cli-user",
        "comments": comments or "",
        "corrections": corrections or {},
    }

    existing_meta = store.load_run_meta(workflow_id)
    if not existing_meta:
        store.save_run_meta(
            workflow_id,
            pdf_blob_name,
            market,
            parent_transaction_id=parent_transaction_id,
        )
        print(f"[RUN_META_CREATED_ON_RESUME] workflow_id={workflow_id}")

    # A user response has been received for this HITL. Mark the run as
    # "HITL Triggered" (same marker the HITL queue path uses) until the
    # resume finishes and _update_run_meta_from_result sets the real
    # outcome (Audit / HITL / Terminated).
    store.update_run_meta_status(workflow_id, "HITL Triggered")

    try:
        result = _validate_workflow_result(
            await service.resume_one(
                checkpoint_id,
                request_id,
                review_response,
            )
        )
        workflow_result_received = True

        result_workflow_id = str(
            result.get("workflow_id") or workflow_id
        ).strip()

        pending_requests = result.get("pending_requests") or []
        if not isinstance(pending_requests, list):
            pending_requests = []

        attached_count = _attach_framework_references_to_hitl_records(
            store=store,
            workflow_id=result_workflow_id,
            result=result,
        )

        run_meta_status = _update_run_meta_from_result(
            store=store,
            workflow_id=result_workflow_id,
            result=result,
            completed_status="Audit",
        )


        if run_meta_status == "Audit":
            print(
                "[AUDIT_STATUS_UPDATED] "
                f"workflow_id={result_workflow_id} "
                "run_meta_status=Audit "
                "queue_message_sent=False"
            )
        elif _send_failed_to_queue_if_needed(
            result=result,
            workflow_id=result_workflow_id,
            pdf_blob_name=pdf_blob_name,
            market=market,
            parent_transaction_id=parent_transaction_id,
        ):
            pass
        elif pending_requests and attached_count != len(pending_requests):
            print(
                "[REVIEW_QUEUE_MESSAGE_NOT_SENT] "
                f"workflow_id={result_workflow_id} "
                f"pending_count={len(pending_requests)} "
                f"attached_count={attached_count} "
                "reason=not_all_hitl_records_enriched"
            )
        # else:
        #     _send_to_review_queue_if_needed(
        #         result=result,
        #         workflow_id=result_workflow_id,
        #         pdf_blob_name=pdf_blob_name,
        #         market=market,
        #         parent_transaction_id=parent_transaction_id,
        #     )

        else:
            termination_message_sent = (
                _send_terminated_to_queue_if_needed(
                    result=result,
                    workflow_id=(
                        result_workflow_id
                    ),
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )
            )

            if not termination_message_sent:
                _send_to_review_queue_if_needed(
                    result=result,
                    workflow_id=(
                        result_workflow_id
                    ),
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )
        _print_result(result_workflow_id, result)
        return result

    except Exception as exc:
        if not workflow_result_received:
            failure_result = {
                "status": "failed",
                "workflow_id": workflow_id,
                "error": str(exc),
            }
            _run_meta_status = _update_run_meta_from_result(
                store=store,
                workflow_id=workflow_id,
                result=failure_result,
            )
            await _finalize_execution_timings(
                store, workflow_id, _run_meta_status
            )
        else:
            # Workflow itself returned, but post-processing failed. Roll
            # the transient "HITL Triggered" marker back to "HITL" so the
            # run is not left stuck in the triggered state.
            try:
                store.update_run_meta_status(workflow_id, "HITL")
            except Exception:
                pass
            print(
                "[POST_WORKFLOW_PROCESSING_FAILED] "
                f"workflow_id={workflow_id} error={exc}"
            )
        raise
    finally:
        await _close_service(service)


async def resume_batch_workflow(
    checkpoint_id: str,
    requests: List[Dict[str, Any]],
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    parent_transaction_id: str = "",
) -> Dict[str, Any]:
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")

    service = WorkflowRunnerService()
    store = WorkflowStore()
    workflow_result_received = False

    existing_meta = store.load_run_meta(workflow_id)
    if not existing_meta:
        store.save_run_meta(
            workflow_id,
            pdf_blob_name,
            market,
            parent_transaction_id=parent_transaction_id,
        )
        print(
            "[RUN_META_CREATED_ON_BATCH_RESUME] "
            f"workflow_id={workflow_id}"
        )

    # A user response has been received for this HITL. Mark the run as
    # "HITL Triggered" (same marker the HITL queue path uses) until the
    # resume finishes and _update_run_meta_from_result sets the real
    # outcome (Audit / HITL / Terminated).
    store.update_run_meta_status(workflow_id, "HITL Triggered")

    try:
        responses: Dict[str, Dict[str, Any]] = {}

        for request in requests:
            if not isinstance(request, dict):
                raise TypeError("Every batch request must be a dictionary")

            request_id = str(request.get("request_id") or "").strip()
            if not request_id:
                raise ValueError(
                    "Every batch request must contain a request_id"
                )

            responses[request_id] = {
                "approved": request.get("approved", True),
                "reviewer": request.get("reviewer", "cli-user"),
                "comments": request.get("comments", ""),
                "corrections": request.get("corrections", {}),
            }

        result = _validate_workflow_result(
            await service.resume(
                checkpoint_id=checkpoint_id,
                responses=responses,
            )
        )
        workflow_result_received = True

        result_workflow_id = str(
            result.get("workflow_id") or workflow_id
        ).strip()

        pending_requests = result.get("pending_requests") or []
        if not isinstance(pending_requests, list):
            pending_requests = []

        attached_count = _attach_framework_references_to_hitl_records(
            store=store,
            workflow_id=result_workflow_id,
            result=result,
        )

        run_meta_status = _update_run_meta_from_result(
            store=store,
            workflow_id=result_workflow_id,
            result=result,
            completed_status="Audit",
        )
        await _finalize_execution_timings(
            store, result_workflow_id, run_meta_status
        )

        if run_meta_status == "Audit":
            print(
                "[AUDIT_STATUS_UPDATED] "
                f"workflow_id={result_workflow_id} "
                "run_meta_status=Audit "
                "queue_message_sent=False"
            )
        elif _send_failed_to_queue_if_needed(
            result=result,
            workflow_id=result_workflow_id,
            pdf_blob_name=pdf_blob_name,
            market=market,
            parent_transaction_id=parent_transaction_id,
        ):
            pass
        elif pending_requests and attached_count != len(pending_requests):
            print(
                "[REVIEW_QUEUE_MESSAGE_NOT_SENT] "
                f"workflow_id={result_workflow_id} "
                f"pending_count={len(pending_requests)} "
                f"attached_count={attached_count} "
                "reason=not_all_hitl_records_enriched"
            )
        # else:
        #     _send_to_review_queue_if_needed(
        #         result=result,
        #         workflow_id=result_workflow_id,
        #         pdf_blob_name=pdf_blob_name,
        #         market=market,
        #         parent_transaction_id=parent_transaction_id,
        #     )

        else:
            termination_message_sent = (
                _send_terminated_to_queue_if_needed(
                    result=result,
                    workflow_id=(
                        result_workflow_id
                    ),
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )
            )

            if not termination_message_sent:
                _send_to_review_queue_if_needed(
                    result=result,
                    workflow_id=(
                        result_workflow_id
                    ),
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )

        _print_result(result_workflow_id, result)
        return result

    except Exception as exc:
        if not workflow_result_received:
            failure_result = {
                "status": "failed",
                "workflow_id": workflow_id,
                "error": str(exc),
            }
            _run_meta_status = _update_run_meta_from_result(
                store=store,
                workflow_id=workflow_id,
                result=failure_result,
            )
            await _finalize_execution_timings(
                store, workflow_id, _run_meta_status
            )
        else:
            # Workflow itself returned, but post-processing failed. Roll
            # the transient "HITL Triggered" marker back to "HITL" so the
            # run is not left stuck in the triggered state.
            try:
                store.update_run_meta_status(workflow_id, "HITL")
            except Exception:
                pass
            print(
                "[POST_WORKFLOW_PROCESSING_FAILED] "
                f"workflow_id={workflow_id} error={exc}"
            )
        raise
    finally:
        await _close_service(service)


async def resume_tool_failure(
    checkpoint_id: str,
    request_id: str,
    action: str,
    workflow_id: str,
    pdf_blob_name: str,
    market: str,
    parent_transaction_id: str = "",
    comments: Optional[str] = None,
) -> Dict[str, Any]:
    service = WorkflowRunnerService()
    store = WorkflowStore()
    workflow_result_received = False

    existing_meta = store.load_run_meta(workflow_id)
    if not existing_meta:
        store.save_run_meta(
            workflow_id,
            pdf_blob_name,
            market,
            parent_transaction_id=parent_transaction_id,
        )
        print(
            "[RUN_META_CREATED_ON_TOOL_RETRY] "
            f"workflow_id={workflow_id}"
        )

    store.update_run_meta_status(workflow_id, "In-Progress")

    tool_failure_response = {
        "action": action,
        "reviewer": "cli-user",
        "comments": comments or "",
    }

    try:
        result = _validate_workflow_result(
            await service.resume_one(
                checkpoint_id,
                request_id,
                tool_failure_response,
            )
        )
        workflow_result_received = True

        result_workflow_id = str(
            result.get("workflow_id") or workflow_id
        ).strip()

        pending_requests = result.get("pending_requests") or []
        if not isinstance(pending_requests, list):
            pending_requests = []

        attached_count = _attach_framework_references_to_hitl_records(
            store=store,
            workflow_id=result_workflow_id,
            result=result,
        )

        run_meta_status = _update_run_meta_from_result(
            store=store,
            workflow_id=result_workflow_id,
            result=result,
            completed_status="Audit",
        )
        await _finalize_execution_timings(
            store, result_workflow_id, run_meta_status
        )

        if run_meta_status == "Audit":
            print(
                "[AUDIT_STATUS_UPDATED] "
                f"workflow_id={result_workflow_id} "
                "run_meta_status=Audit "
                "queue_message_sent=False"
            )
        elif _send_failed_to_queue_if_needed(
            result=result,
            workflow_id=result_workflow_id,
            pdf_blob_name=pdf_blob_name,
            market=market,
            parent_transaction_id=parent_transaction_id,
        ):
            pass
        elif pending_requests and attached_count != len(pending_requests):
            print(
                "[REVIEW_QUEUE_MESSAGE_NOT_SENT] "
                f"workflow_id={result_workflow_id} "
                f"pending_count={len(pending_requests)} "
                f"attached_count={attached_count} "
                "reason=not_all_hitl_records_enriched"
            )
        # else:
        #     _send_to_review_queue_if_needed(
        #         result=result,
        #         workflow_id=result_workflow_id,
        #         pdf_blob_name=pdf_blob_name,
        #         market=market,
        #         parent_transaction_id=parent_transaction_id,
        #     )

        else:
            termination_message_sent = (
                _send_terminated_to_queue_if_needed(
                    result=result,
                    workflow_id=(
                        result_workflow_id
                    ),
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )
            )

            if not termination_message_sent:
                _send_to_review_queue_if_needed(
                    result=result,
                    workflow_id=(
                        result_workflow_id
                    ),
                    pdf_blob_name=pdf_blob_name,
                    market=market,
                    parent_transaction_id=(
                        parent_transaction_id
                    ),
                )

        _print_result(result_workflow_id, result)
        return result

    except Exception as exc:
        if not workflow_result_received:
            failure_result = {
                "status": "failed",
                "workflow_id": workflow_id,
                "error": str(exc),
            }
            _run_meta_status = _update_run_meta_from_result(
                store=store,
                workflow_id=workflow_id,
                result=failure_result,
            )
            await _finalize_execution_timings(
                store, workflow_id, _run_meta_status
            )
        else:
            print(
                "[POST_WORKFLOW_PROCESSING_FAILED] "
                f"workflow_id={workflow_id} error={exc}"
            )
        raise
    finally:
        await _close_service(service)


async def main(
    mode: str,
    workflow_id: Optional[str] = None,
    pdf_blob_name: Optional[str] = None,
    market: Optional[str] = None,
    parent_transaction_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    request_id: Optional[str] = None,
    approved: Optional[bool] = None,
    comments: Optional[str] = None,
    corrections: Optional[Dict[str, Any]] = None,
    action: Optional[str] = None,
    requests: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run start, resume, batchresume, or retry mode."""
    normalized_mode = str(mode or "").strip().lower()
    normalized_workflow_id = _validate_required(workflow_id, "workflow_id")
    normalized_pdf_blob_name = _validate_required(
        pdf_blob_name,
        "pdf_blob_name",
    )
    normalized_market = _validate_required(market, "market")
    normalized_parent_id = str(parent_transaction_id or "").strip()

    if normalized_mode == "start":
        return await start_workflow(
            workflow_id=normalized_workflow_id,
            pdf_blob_name=normalized_pdf_blob_name,
            market=normalized_market,
            parent_transaction_id=normalized_parent_id,
        )

    if normalized_mode == "resume":
        normalized_checkpoint_id = _validate_required(
            checkpoint_id,
            "checkpoint_id",
        )
        normalized_request_id = _validate_required(
            request_id,
            "request_id",
        )

        return await resume_workflow(
            checkpoint_id=normalized_checkpoint_id,
            request_id=normalized_request_id,
            approved=True if approved is None else bool(approved),
            workflow_id=normalized_workflow_id,
            pdf_blob_name=normalized_pdf_blob_name,
            market=normalized_market,
            parent_transaction_id=normalized_parent_id,
            corrections=corrections,
            comments=comments,
        )

    if normalized_mode == "batchresume":
        normalized_checkpoint_id = _validate_required(
            checkpoint_id,
            "checkpoint_id",
        )

        return await resume_batch_workflow(
            checkpoint_id=normalized_checkpoint_id,
            requests=requests or [],
            workflow_id=normalized_workflow_id,
            pdf_blob_name=normalized_pdf_blob_name,
            market=normalized_market,
            parent_transaction_id=normalized_parent_id,
        )

    if normalized_mode == "retry":
        normalized_checkpoint_id = _validate_required(
            checkpoint_id,
            "checkpoint_id",
        )
        normalized_request_id = _validate_required(
            request_id,
            "request_id",
        )
        normalized_action = _validate_required(action, "action")

        return await resume_tool_failure(
            checkpoint_id=normalized_checkpoint_id,
            request_id=normalized_request_id,
            action=normalized_action,
            workflow_id=normalized_workflow_id,
            pdf_blob_name=normalized_pdf_blob_name,
            market=normalized_market,
            parent_transaction_id=normalized_parent_id,
            comments=comments,
        )

    raise ValueError(
        f"Unsupported mode '{mode}'. "
        "Use start, resume, batchresume, or retry."
    )


if __name__ == "__main__":
    MODE = "start"
    # MODE = "resume"
    # MODE = "batchresume"
    # # MODE = "retry"

    PDF_BLOB_NAME = "GM Backup June19.pdf"
    MARKET = "CAN"
    if MODE == "start":
        workflow_id = str(uuid.uuid4())
        parent_transaction_id = str(
            uuid.uuid4()
        )

        asyncio.run(start_workflow(
                workflow_id=workflow_id,
                pdf_blob_name=PDF_BLOB_NAME,
                market=MARKET,
                parent_transaction_id=(
                    parent_transaction_id
                ),
            )
        )
    elif MODE == "batchresume":
        asyncio.run(
            resume_batch_workflow(
                checkpoint_id="6450ca73-a210-4059-926d-0588f895c4e5",
                requests=[
                    {
                        "request_id": (
                            "1b5c555b-9d1d-4e06-a71a-ceaf3ba5cf4a"
                        ),
                        "approved": True,
                        "comments": (
                            "Approved tab '1 ser_EL_EP' "
                            "for invoice 151753350."
                        ),
                        "corrections": {
                            "tab_name": "1 ser_EL_EP",
                        },
                    },
                    {
                        "request_id": (
                            "f00b899f-2ec5-4a7d-836c-f0e012756d7e"
                        ),
                        "approved": True,
                        "comments": (
                            "Approved tab '1 ser_EL_EP' "
                            "for invoice 151753349."
                        ),
                        "corrections": {
                            "tab_name": "1 ser_EL_EP",
                        },
                    },
                ],
                workflow_id="becd5bc6-b63b-4c48-8676-5896b73c1e00",
                pdf_blob_name="Miscellaneous charges.pdf",
                market="CAN",
                parent_transaction_id=(
                    "076b55da-3435-4cff-82ca-a137e62670b3"
                ),
            )
        )
    elif MODE == "resume":
        asyncio.run(
            resume_workflow(
                checkpoint_id="d7aa4f91-5487-42a6-9589-6aae5886f0a8",
                request_id="4dc0cac4-4f1b-4e2c-b6d6-718b423c475d",
                approved=True,
                workflow_id="165ab6fc-381a-4320-99ef-3b13ea88e2ad",
                pdf_blob_name="Sobeys Vendor Billing Coupons.pdf",
                market="CAN",
                parent_transaction_id="c65c3501-f005-49df-8862-88389a744ff3",
                corrections={
                    "selected_vendor": "SOBEYS QUEBEC INC",
                },
                comments="Approved vendor selection 'SOBEYS QUEBEC INC' as-is during vendor_resolution review.",
            )
        )
    elif MODE == "retry":
        asyncio.run(resume_tool_failure(
                checkpoint_id="checkpoint-id",
                request_id="request-id",
                action="retry",
                workflow_id=(
                    "existing-workflow-id"
                ),
                pdf_blob_name=PDF_BLOB_NAME,
                market=MARKET,
                parent_transaction_id=(
                    "existing-parent-transaction-id"
                ),
            )
        )


