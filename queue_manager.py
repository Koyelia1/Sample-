import base64
import logging
import os

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from azure.core.pipeline.transport import RequestsTransport
from azure.storage.queue import QueueClient
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential

transport = RequestsTransport(verify=False)
logger = logging.getLogger(__name__)

load_dotenv()


class QueueStorageManager:
    """Manages operations with Azure Queue Storage."""

    def __init__(self):
        self.review_queue_name = os.getenv("AZURE_STORAGE_REVIEW_QUEUE")
        self.eml_queue_name = os.getenv("AZURE_STORAGE_QUEUE_NAME")

        credentials = DefaultAzureCredential()
        self.review_queue_client = QueueClient(
            account_url=os.environ.get("QUEUE_ACCOUNT_URL"),
            queue_name=self.review_queue_name,
            credential=credentials,
            connection_verify=False
        )

        self.queue_client = QueueClient(
            account_url=os.environ.get("QUEUE_ACCOUNT_URL"),
            queue_name=self.eml_queue_name,
            credential=credentials
        )

    def send_message(self, message: str) -> None:
        """Send a base64 encoded message to the queue."""
        try:
            message_bytes = message.encode("utf-8")
            base64_message = base64.b64encode(message_bytes).decode("utf-8")
            self.queue_client.send_message(base64_message)
            logger.info("Message sent to queue successfully")
        except Exception as e:
            logger.error(f"Failed to send message to queue: {str(e)}")
            raise

    def send_message_review_queue(self, message: str) -> None:
        """Send a base64 encoded message to the review assignment queue."""
        try:
            message_bytes = message.encode("utf-8")
            base64_message = base64.b64encode(message_bytes).decode("utf-8")
            self.review_queue_client.send_message(base64_message)
            logger.info("Message sent to review queue successfully")
            print(f"Message sent to review queue successfully:", message)
        except Exception as e:
            logger.error(f"Failed to send message to review queue: {str(e)}")
            raise


# Create global queue storage manager instance
queue_manager = QueueStorageManager()
