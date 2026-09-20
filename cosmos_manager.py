import logging
import os

from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
from utility.cosmos_json_store import load_secrets
from utility.time_formatter import now_iso
from azure.identity import DefaultAzureCredential

load_dotenv()
logger = logging.getLogger(__name__)


class CosmosManager:
    """Manages operations with Azure Cosmos DB."""

    def __init__(self):
        self.jobs_container_name = os.getenv("COSMOS_EML_METADATA_CONTAINER")
        secrets = load_secrets()
        self.endpoint = secrets.get("cosmos_endpoint")
        self.database_name = secrets.get("cosmos_database")

        if not self.endpoint:
            error_msg = f"[COSMOS_INIT] ❌ COSMOS NOT INITIALIZED - Missing: Endpoint={bool(self.endpoint)}"
            print(error_msg)
            logger.error(error_msg)
            self.jobs_container = None
            self.users_container = None
        else:
            try:
                self.client = self._create_client()
                database = self.client.create_database_if_not_exists(
                    id=self.database_name
                )

                self.jobs_container = database.create_container_if_not_exists(
                    id=self.jobs_container_name,
                    partition_key=PartitionKey(path="/parent_transaction_id"),
                )
                print("[COSMOS_INIT] ✓ Successfully initialized jobs container.")
                logger.info("[COSMOS_INIT] Successfully initialized jobs container.")

                users_container_name = os.getenv("COSMOS_USERS_CONTAINER", "users")
                self.users_container = database.create_container_if_not_exists(
                    id=users_container_name,
                    partition_key=PartitionKey(path="/UserId"),
                )
                print("[COSMOS_INIT] ✓ Successfully initialized users container.")
                logger.info("[COSMOS_INIT] Successfully initialized users container.")

                self.workflow_container_name = os.getenv("COSMOS_WORKFLOW_EXECUTION_CONTAINER","workflow_execution_store")
                self.workflow_container = database.create_container_if_not_exists(
                    id = self.workflow_container_name,
                    partition_key=PartitionKey(path="/workflow_id"),
                )

            except Exception as e:
                error_msg = (
                    f"[COSMOS_INIT] ❌ Failed to initialize Cosmos containers: {str(e)}"
                )
                print(error_msg)
                logger.error(error_msg)
                self.jobs_container = None
                self.users_container = None

    def _create_client(self) -> CosmosClient:
        try:
            credentials = DefaultAzureCredential()
            return CosmosClient(self.endpoint, credential=credentials)
        except Exception as e:
            logger.error(f"Failed to create Cosmos Client: {str(e)}")
            raise

    def create_job_entry(
        self,
        parent_transaction_id: str,
        blob_path: str,
        file_name: str,
        status: str = "Queued",
    ) -> dict:
        """Create a new job entry in Cosmos DB."""
        if not self.jobs_container:
            return {"id": parent_transaction_id, "status": status}

        try:
            processing_name = f"{parent_transaction_id}_{file_name}"
            item = {
                "id": parent_transaction_id,
                "parent_transaction_id": parent_transaction_id,
                "OriginaFileName": file_name,
                "FileName": processing_name,
                "BlobPath": blob_path,
                "status": status,
                "created_at": now_iso(),
                "type": "metadata",
            }
            self.jobs_container.create_item(body=item)
            logger.info(f"Cosmos entry created for transaction: {parent_transaction_id}")
            return item
        except Exception as e:
            logger.error(f"Failed to create cosmos entry: {str(e)}")
            raise

    def update_job_status(
        self, parent_transaction_id: str, status: str, result_data: dict = None
    ):
        """Update the status of a job in Cosmos DB."""
        if not self.jobs_container:
            error_msg = f"[COSMOS_UPDATE] ❌ SKIPPING status update for {parent_transaction_id} → {status}: Container not initialized"
            print(error_msg)
            logger.error(
                f"[COSMOS_UPDATE] SKIPPING status update for {parent_transaction_id} → {status}: Cosmos container not initialized (env vars missing)"
            )
            return

        try:
            item = self.jobs_container.read_item(
                item=parent_transaction_id, partition_key=parent_transaction_id
            )
            item["status"] = status
            item["LastUpdated"] = now_iso()
            if result_data is not None:
                item["result_data"] = result_data
            self.jobs_container.replace_item(item=parent_transaction_id, body=item)
            success_msg = (
                f"[COSMOS_UPDATE] ✓ Updated job {parent_transaction_id} to status: {status}"
            )
            print(success_msg)
            logger.info(success_msg)
        except Exception as e:
            error_msg = (
                f"[COSMOS_UPDATE] ❌ Failed to update job {parent_transaction_id}: {str(e)}"
            )
            print(error_msg)
            logger.error(
                f"[COSMOS_UPDATE] Failed to update job status for {parent_transaction_id}: {str(e)}"
            )
            raise

    def create_inputs_job_entry(
        self,
        attachment_id: str,
        parent_transaction_id: str,
        blob_path: str,
        file_name: str,
        status: str = "Queued",
    ) -> dict:
        """Create a new job entry in Cosmos DB."""
        if not self.jobs_container:
            return {"id": attachment_id, "status": status}

        try:
            processing_name = f"{attachment_id}_{file_name}"
            item = {
                "id": attachment_id,
                "attachment_id": attachment_id,
                "parent_transaction_id": parent_transaction_id,
                "OriginaFileName": file_name,
                "FileName": processing_name,
                "BlobPath": blob_path,
                "status": status,
                "created_at": now_iso(),
                "type": "metadata",
            }
            self.input_container.create_item(body=item)
            logger.info(f"Cosmos entry created for transaction: {attachment_id}")
            return item
        except Exception as e:
            logger.error(f"Failed to create cosmos entry: {str(e)}")
            raise

    def update_workflow_execution_status(
            self,attachment_id:str,status:str,result_data:dict = None
    ):
        if not self.workflow_container:
            error_msg = f"[WORKFLOW_UPDATE] SKIPPING status update for {attachment_id} → {status}: Workflow container not initialized"
            print(error_msg)
            logger.error(error_msg)
            return
        try:
            item = {
                "id" : attachment_id,
                "attachment_id": attachment_id,
                "status": status,
                "LastUpdated": now_iso(),
            }
            if result_data is not None:
                item["result_data"] = result_data
            self.workflow_container.upsert_item(body=item)
            success_msg = f"[WORKFLOW_UPDATE] Updated workflow execution {attachment_id} to status: {status}"
            print(success_msg)
            logger.info(success_msg)
        except Exception as e:
            error_msg = f"[WORKFLOW_UPDATE] Failed to update workflow execution {attachment_id}: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            raise

# Create global cosmos manager instance
cosmos_manager = CosmosManager()

# cosmos_manager.update_job_status("486f19695647","Processing")