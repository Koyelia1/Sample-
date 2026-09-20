from agent_framework.exceptions import logger
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import base64
import json
import logging
import os
import time
from azure.storage.queue import QueueClient
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from cosmos_manager import cosmos_manager
from main_wf import main
from queue_manager import queue_manager
from typing import Any, Dict

load_dotenv()

logging.basicConfig(level=logging.INFO)

QUEUE_NAME = os.getenv("AZURE_STORAGE_QUEUE_NAME")


def _map_result_to_status(result: Dict[str, Any]) -> str:
    status = (result or {}).get("status", "")
    logger.info(f"Mapping backend status '{status}' to user-facing status")
    print(f"Mapping backend status '{status}' to user-facing status")

    if status in {"paused", "pending_review", "stale_token"}:
        return "HITL"
    elif status in {"failed", "rejected"}:
        return "Terminated"
    elif status in {"completed", "completed_with_rejections" , "query_raised"}:
        return "Completed"
    else:
        return "In-Progress"


def process_message(msg):
    try:
        message_body = base64.b64decode(msg.content).decode("utf-8")
        message_data = json.loads(message_body)
        #message_data={"parent_transaction_id": "ee1c937bd44c", "attachment_id": "ee1c937bd44c", "file_name": "ee1c937bd44c_Albertsons companies-Sample.pdf", "market": "USR"}
        # message_data={"parent_transaction_id": "97363bb57ba2", "attachment_id": "97363bb57ba2", "file_name": "97363bb57ba2_Albertsons companies-Sample.pdf", "market": "USR"}
        parent_transaction_id = message_data.get("parent_transaction_id")
        attachment_id = message_data.get("attachment_id")
        blob_name = message_data.get("file_name")
        updated_blob_name = blob_name
        market_name = message_data.get("market")

        logging.info(f"Received message for transaction_id: {attachment_id}, blob_name: {blob_name}")
        if not attachment_id or not blob_name:
            logging.error(f"Invalid message: {message_data}")
            return


        logging.info(f"Processing {attachment_id}")

        # update processing
        cosmos_manager.update_job_status(parent_transaction_id, status="Processing")
        cosmos_manager.update_workflow_execution_status(attachment_id, status="Processing")
        logger.info(f"Updated workflow execution status to 'Processing' for transaction_id={attachment_id}")
        # business logic
  
        # workflow_id = f"wf-{uuid.uuid4()}"
        logger.info(f"Starting main workflow executor for transaction_id={attachment_id}, blob_name={blob_name}, market_name={market_name}")
        print(f"Starting main workflow executor for transaction_id={attachment_id}, blob_name={blob_name}, market_name={market_name}")
        # json_result = main(updated_blob_name,transaction_id,market_name)
        import asyncio

       
        json_result = asyncio.run(
            main(
                mode="start",
                pdf_blob_name=updated_blob_name,
                workflow_id=attachment_id,
                market=market_name,
                parent_transaction_id=parent_transaction_id,
            )
        )

        logger.info(f"Completed main workflow executor for transaction_id={attachment_id}, blob_name={blob_name}, market_name={market_name} with result: {json_result}")
        print(f"Completed main workflow executor for transaction_id={attachment_id}, blob_name={blob_name}, market_name={market_name} with result: {json_result}")
    
        backend_status = _map_result_to_status(json_result)

        logger.info(f"Mapped backend status: {backend_status}")
        print(f"Mapped backend status: {backend_status}")
        if backend_status == "Completed":
            queue_manager.send_message_review_queue(
                json.dumps(
                    {
                        "attachment_id": attachment_id,
                        "file_path": blob_name,
                        "parent_transaction_id": parent_transaction_id,
                        "market": market_name,
                        "status": "completed",
                    }
                )
            )
    except Exception as e:
        logging.error(f"Worker failed: {str(e)}", exc_info=True)


def run_worker():

    print("Worker started, listening for messages...")
    credentials = DefaultAzureCredential()
    queue_client = QueueClient(
        account_url=os.environ.get("QUEUE_ACCOUNT_URL"),
        credential=credentials,
        queue_name=QUEUE_NAME,
        message_decode_policy=None,  # Get raw bytes to handle decoding ourselves
        message_encode_policy=None,  # Disable encoding to get raw bytes
        connection_verify=False,
    )

    while True:
        print("Checking for messages...")
        messages = list(
            queue_client.receive_messages(messages_per_page=1, visibility_timeout=1200)
        )

        for msg in messages:
            print(f"Received message: {msg.id}")
            try:
                process_message(msg)
            finally:
                try:
                    queue_client.delete_message(msg.id, msg.pop_receipt)
                    print(f"Processed and deleted message: {msg.id}")
                except Exception as e:
                    logging.error(
                        f"Failed to delete message {msg.id}: {str(e)}", exc_info=True
                    )

        time.sleep(2)  # avoid tight loop


if __name__ == "__main__":
    run_worker()
