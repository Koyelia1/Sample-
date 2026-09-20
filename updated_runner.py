from typing import Any, Dict, List, Optional


class WorkflowRunnerService:
    """
    Single service for workflow start, pause detection, and resume.

    Backend should call:
        - start(payload)
        - resume(checkpoint_id, responses)

    This service works with Microsoft Agent Framework workflows using:
        - ctx.request_info(...)
        - @response_handler
        - CosmosCheckpointStorage / FileCheckpointStorage
    """

    def __init__(
        self,
        workflow,
        checkpoint_storage,
    ):
        """
        Args:
            workflow:
                Built Agent Framework workflow.

            checkpoint_storage:
                CosmosCheckpointStorage or any CheckpointStorage implementation.
        """

        self.workflow = workflow
        self.checkpoint_storage = checkpoint_storage

    async def start(
        self,
        payload: Any,
    ) -> Dict[str, Any]:
        """
        Start a new workflow run.

        Returns either:
            {
                "status": "completed",
                "outputs": [...]
            }

        or:
            {
                "status": "pending_review",
                "workflow_name": "...",
                "checkpoint_id": "...",
                "pending_requests": [...]
            }
        """

        stream = self.workflow.run(
            message=payload,
            stream=True,
        )

        return await self._consume_stream(stream)

    async def resume(
        self,
        checkpoint_id: str,
        responses: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Resume workflow from checkpoint.

        Args:
            checkpoint_id:
                Checkpoint id stored when HITL was raised.

            responses:
                Dictionary where key is request_id and value is reviewer response.

                Example:
                {
                    "request_1": {
                        "approved": false,
                        "reviewer": "john.doe",
                        "comments": "Corrected vendor",
                        "corrections": {
                            "selected_vendor": "Microsoft"
                        }
                    }
                }

        Returns either completed output or more pending HITL requests.
        """

        stream = self.workflow.run(
            checkpoint_id=checkpoint_id,
            responses=responses,
            stream=True,
        )

        return await self._consume_stream(stream)

    async def resume_one(
        self,
        checkpoint_id: str,
        request_id: str,
        review_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convenience function for backend when resolving one HITL request.
        """

        return await self.resume(
            checkpoint_id=checkpoint_id,
            responses={
                request_id: review_response,
            },
        )

    async def _consume_stream(
        self,
        stream,
    ) -> Dict[str, Any]:
        """
        Internal stream processor.

        Collects all request_info events before returning.
        This supports multiple HITL requests appearing at the same time.
        """

        pending_requests: List[Dict[str, Any]] = []
        outputs: List[Any] = []
        errors: List[Any] = []

        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "request_info":
                pending_requests.append(
                    {
                        "request_id": getattr(
                            event,
                            "request_id",
                            None,
                        ),
                        "review_payload": getattr(
                            event,
                            "data",
                            None,
                        ),
                    }
                )

            elif event_type == "output":
                outputs.append(
                    getattr(
                        event,
                        "data",
                        None,
                    )
                )

            elif event_type == "error":
                errors.append(
                    getattr(
                        event,
                        "data",
                        event,
                    )
                )

        if errors:
            return {
                "status": "failed",
                "workflow_name": self.workflow.name,
                "errors": errors,
            }

        if pending_requests:
            checkpoint_id = await self._get_latest_checkpoint_id()

            return {
                "status": "pending_review",
                "workflow_name": self.workflow.name,
                "checkpoint_id": checkpoint_id,
                "pending_requests": pending_requests,
            }

        return {
            "status": "completed",
            "workflow_name": self.workflow.name,
            "outputs": outputs,
        }

    async def _get_latest_checkpoint_id(
        self ,) -> Optional[str]:
        """
        Gets latest checkpoint id for the workflow.

        CosmosCheckpointStorage supports get_latest(workflow_name=...).
        """

        latest = await self.checkpoint_storage.get_latest(
            workflow_name=self.workflow.name,
        )

        if latest is None:
            return None

        return latest.checkpoint_id