from agent_framework.exceptions import logger
import urllib3

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict

from azure.identity import DefaultAzureCredential
from azure.storage.queue import QueueClient
from dotenv import load_dotenv

from cosmos_manager import cosmos_manager
from main_wf import main


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

QUEUE_NAME = "hitl-processing-queue"


def decode_queue_message(content: Any) -> Dict[str, Any]:
    """
    Decode an Azure Storage Queue message.

    Supports:
    1. Base64-encoded JSON
    2. Plain JSON
    3. Bytes containing either format
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    if not isinstance(content, str):
        raise TypeError(
            "Queue message content must be a string or bytes. "
            f"Received: {type(content).__name__}"
        )

    content = content.strip()

    if not content:
        raise ValueError("Queue message content is empty")

    # First try plain JSON.
    try:
        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Queue message JSON must be an object"
            )

        return parsed

    except json.JSONDecodeError:
        pass

    # If it is not plain JSON, try Base64 JSON.
    try:
        decoded_content = base64.b64decode(
            content,
            validate=True,
        ).decode("utf-8")

        parsed = json.loads(decoded_content)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Decoded queue message JSON must be an object"
            )

        return parsed

    except Exception as exc:
        raise ValueError(
            "Queue message is neither plain JSON nor "
            "valid Base64-encoded JSON"
        ) from exc


def payload_list_to_dictionary(
    payload: Any,
) -> Dict[str, Any]:
    """
    Convert:

    [
        {"name": "tab_name", "value": "Scan"},
        {"name": "market", "value": "USR"}
    ]

    into:

    {
        "tab_name": "Scan",
        "market": "USR"
    }
    """
    if payload is None:
        return {}

    if isinstance(payload, dict):
        return payload

    if not isinstance(payload, list):
        raise TypeError(
            "payload must be a list or dictionary"
        )

    corrections: Dict[str, Any] = {}

    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(
                f"payload[{index}] must be an object"
            )

        name = str(
            item.get("name") or ""
        ).strip()

        if not name:
            raise ValueError(
                f"payload[{index}].name is missing"
            )

        corrections[name] = item.get("value")

    return corrections


def get_resume_items(
    message_data: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """
    Read and normalize resume items from both supported properties:

    1. "payloads"
    2. "documents"

    Both checkpoint field names are supported:

    1. "checkpoint_id"
    2. "checkpoint"

    If both "payloads" and "documents" are provided, items from both
    arrays are included.
    """
    resume_items: list[Dict[str, Any]] = []

    for property_name in (
        "payloads",
        "documents",
    ):
        items = message_data.get(property_name)

        if items is None:
            continue

        if not isinstance(items, list):
            raise TypeError(
                f"'{property_name}' must be a JSON array"
            )

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise TypeError(
                    f"{property_name}[{index}] "
                    "must be an object"
                )

            normalized_item = dict(item)

            checkpoint_id = str(
                normalized_item.get("checkpoint_id")
                or normalized_item.get("checkpoint")
                or ""
            ).strip()

            if not checkpoint_id:
                raise ValueError(
                    f"{property_name}[{index}] must "
                    "contain 'checkpoint_id' or "
                    "'checkpoint'"
                )

            normalized_item["checkpoint_id"] = (
                checkpoint_id
            )

            resume_items.append(normalized_item)

    if not resume_items:
        raise ValueError(
            "The queue message must contain a non-empty "
            "'payloads' or 'documents' array. "
            f"Available properties: "
            f"{list(message_data.keys())}"
        )

    return resume_items


def validate_common_fields(
    message_data: Dict[str, Any],
) -> tuple[str, str, str, str]:
    workflow_id = str(
        message_data.get("workflow_id") or ""
    ).strip()

    parent_transaction_id = str(
        message_data.get(
            "parent_transaction_id"
        )
        or ""
    ).strip()

    market = str(
        message_data.get("market") or ""
    ).strip()

    pdf_blob_name = str(
        message_data.get("pdf_blob_name")
        or message_data.get("file_path")
        or ""
    ).strip()

    missing_fields = []

    if not workflow_id:
        missing_fields.append("workflow_id")

    if not pdf_blob_name:
        missing_fields.append("pdf_blob_name")

    if not market:
        missing_fields.append("market")

    if missing_fields:
        raise ValueError(
            "Missing required field(s): "
            + ", ".join(missing_fields)
        )

    return (
        workflow_id,
        parent_transaction_id,
        market,
        pdf_blob_name,
    )


def process_message(msg) -> bool:
    """
    Process one queue message.

    Returns True only when processing succeeds.
    The caller should delete the queue message only when True.
    """
    print(
        "\n"
        "============================================"
    )
    print(
        f"[HITL] Processing message id={msg.id}"
    )
    print(
        "============================================"
    )

    try:
        message_data = decode_queue_message(
            msg.content
        )

        print("[HITL] Decoded queue message:")
        print(
            json.dumps(
                message_data,
                indent=2,
                default=str,
            )
        )

        (
            workflow_id,
            parent_transaction_id,
            market,
            pdf_blob_name,
        ) = validate_common_fields(
            message_data
        )

        # Supports both "payloads" and "documents".
        # Also normalizes "checkpoint" to "checkpoint_id".
        payloads = get_resume_items(
            message_data
        )

        print(
            "[HITL] Resume items found: "
            f"{len(payloads)}"
        )

        cosmos_manager.update_workflow_execution_status(
            workflow_id,
            status="Processing",
        )

        # NOTE: the run_meta "HITL Triggered" marker (and its rollback on
        # failure) is handled inside main_wf.resume_workflow /
        # resume_batch_workflow, so it is intentionally not set here.

        # Group payloads by checkpoint because batch resume can
        # only resume request IDs belonging to one checkpoint.
        payloads_by_checkpoint: Dict[
            str,
            list[Dict[str, Any]],
        ] = {}

        for index, payload_item in enumerate(
            payloads
        ):
            if not isinstance(payload_item, dict):
                raise TypeError(
                    f"payloads[{index}] must be an object"
                )

            checkpoint_id = str(
                payload_item.get("checkpoint_id")
                or ""
            ).strip()

            request_id = str(
                payload_item.get("request_id")
                or ""
            ).strip()

            if not checkpoint_id:
                raise ValueError(
                    f"payloads[{index}].checkpoint_id "
                    "is missing"
                )

            if not request_id:
                raise ValueError(
                    f"payloads[{index}].request_id "
                    "is missing"
                )

            payloads_by_checkpoint.setdefault(
                checkpoint_id,
                [],
            ).append(payload_item)

        print(
            "[HITL] Checkpoint groups found: "
            f"{len(payloads_by_checkpoint)}"
        )

        results = []

        for (
            checkpoint_id,
            checkpoint_payloads,
        ) in payloads_by_checkpoint.items():

            # One request for this checkpoint: resume.
            if len(checkpoint_payloads) == 1:
                payload_item = checkpoint_payloads[0]

                request_id = str(
                    payload_item["request_id"]
                ).strip()

                corrections = (
                    payload_list_to_dictionary(
                        payload_item.get("payload", [])
                    )
                )

                approved = payload_item.get(
                    "approved",
                    True,
                )

                comments = str(
                    payload_item.get("comments")
                    or "Processed from HITL queue"
                )

                print(
                    "[HITL] Calling main in resume mode: "
                    f"workflow_id={workflow_id}, "
                    f"checkpoint_id={checkpoint_id}, "
                    f"request_id={request_id}"
                )

                result = asyncio.run(
                    main(
                        mode="resume",
                        workflow_id=workflow_id,
                        pdf_blob_name=pdf_blob_name,
                        market=market,
                        parent_transaction_id=(
                            parent_transaction_id
                        ),
                        checkpoint_id=checkpoint_id,
                        request_id=request_id,
                        approved=bool(approved),
                        comments=comments,
                        corrections=corrections,
                    )
                )

                results.append(result)

            # Multiple requests for this checkpoint:
            # batchresume.
            else:
                requests = []

                for payload_item in (
                    checkpoint_payloads
                ):
                    corrections = (
                        payload_list_to_dictionary(
                            payload_item.get(
                                "payload",
                                [],
                            )
                        )
                    )

                    requests.append(
                        {
                            "request_id": str(
                                payload_item[
                                    "request_id"
                                ]
                            ).strip(),
                            "approved": (
                                payload_item.get(
                                    "approved",
                                    True,
                                )
                            ),
                            "reviewer": (
                                payload_item.get(
                                    "reviewer",
                                    "hitl-queue",
                                )
                            ),
                            "comments": (
                                payload_item.get(
                                    "comments",
                                    "Processed from "
                                    "HITL queue",
                                )
                            ),
                            "corrections": corrections,
                        }
                    )

                print(
                    "[HITL] Calling main in "
                    "batchresume mode: "
                    f"workflow_id={workflow_id}, "
                    f"checkpoint_id={checkpoint_id}, "
                    f"request_count={len(requests)}"
                )

                result = asyncio.run(
                    main(
                        mode="batchresume",
                        workflow_id=workflow_id,
                        pdf_blob_name=pdf_blob_name,
                        market=market,
                        parent_transaction_id=(
                            parent_transaction_id
                        ),
                        checkpoint_id=checkpoint_id,
                        requests=requests,
                    )
                )

                results.append(result)

        print(
            "[HITL] Message processed successfully: "
            f"id={msg.id}"
        )
        print(
            "[HITL] Results: "
            f"{json.dumps(results, default=str)}"
        )

        return True

    except Exception as exc:
        logger.error(
            "HITL processing failed for "
            f"message id={msg.id}: {exc}",
            exc_info=True,
        )

        print(
            "[HITL] Message was NOT deleted because "
            "processing failed."
        )

        return False


def run_worker() -> None:
    account_url = os.environ.get(
        "QUEUE_ACCOUNT_URL"
    )

    if not account_url:
        raise ValueError(
            "QUEUE_ACCOUNT_URL environment variable "
            "is missing"
        )

    print(
        "\n"
        "============================================"
    )
    print("[HITL] Worker started")
    print(f"[HITL] Queue name : {QUEUE_NAME}")
    print(f"[HITL] Account URL: {account_url}")
    print(
        "============================================\n"
    )

    credentials = DefaultAzureCredential()

    queue_client = QueueClient(
        account_url=account_url,
        credential=credentials,
        queue_name=QUEUE_NAME,
        message_decode_policy=None,
        message_encode_policy=None,
        connection_verify=False,
    )

    while True:
        try:
            print(
                "[HITL] Checking for messages..."
            )

            messages = list(
                queue_client.receive_messages(
                    messages_per_page=1,
                    visibility_timeout=1200,
                )
            )

            if not messages:
                print("[HITL] No messages found")

            for msg in messages:
                print(
                    "[HITL] Message received: "
                    f"id={msg.id}"
                )

                processing_succeeded = (
                    process_message(msg)
                )

                if processing_succeeded:
                    try:
                        queue_client.delete_message(
                            msg.id,
                            msg.pop_receipt,
                        )

                        print(
                            "[HITL] Processed and deleted "
                            f"message: {msg.id}"
                        )

                    except Exception as delete_error:
                        logger.error(
                            "Failed to delete message "
                            f"{msg.id}: {delete_error}",
                            exc_info=True,
                        )
                else:
                    print(
                        "[HITL] Processing failed. "
                        "Message retained for retry: "
                        f"{msg.id}"
                    )

            time.sleep(2)

        except KeyboardInterrupt:
            print("\n[HITL] Worker stopped by user")
            break

        except Exception as worker_error:
            logger.error(
                f"HITL worker loop failed: "
                f"{worker_error}",
                exc_info=True,
            )

            time.sleep(5)


if __name__ == "__main__":
    run_worker()