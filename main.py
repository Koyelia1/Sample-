import asyncio
import uuid
from agent_framework import (
    WorkflowBuilder,
    WorkflowContext,
    executor,
    FileCheckpointStorage
)

from telemetry.otel import tracer  # ✅ ADDED

from agents_main.classification_agent import build_agent as build_classification_agent
from agents_main.indexing_agent import _build_single_invoice_agent as build_indexing_agent


"""
Simple Workflow:
run_classification → run_parallel_indexing
"""


# Step 1: Classification
import json
import re


@executor(id="run_classification")
async def run_classification(
    ctx_input: dict,
    ctx: WorkflowContext[dict]
):
    #with tracer.start_as_current_span("run_classification"):  # ✅ ADDED
    workflow_id=ctx_input["workflow_id"]
    with tracer.start_as_current_span("Classification_agent") as span:
        span.add_event(
            name="Classification started",
            attributes={
            "workflow_id": workflow_id,  # ✅ UPDATED (dynamic)
            "input": json.dumps(ctx_input),
    }
        )
        agent = build_classification_agent()

        try:
            file_name = ctx_input["pdf_blob_name"]
            market = ctx_input["market"]

            input_text = f"""
            pdf_blob_name: {file_name}
            market: {market}
            """

            result = await agent.run(input_text)

            print("Classification output:", result)

            parsed = {}
            try:
                text = str(result)
                match = re.search(r"\{.*\}", text, re.DOTALL)

                if match:
                    parsed = json.loads(match.group())
                else:
                    parsed = {"error": "no_json_found", "raw": text}

            except Exception as e:
                parsed = {"error": str(e), "raw": str(result)}

        except Exception as e:
            parsed = {"error": str(e)}

        ctx_input["classification"] = parsed
        span.add_event(
            name="Classification Completed",
            attributes={
            "workflow_id": workflow_id,  # ✅ UPDATED (dynamic)
            "output": json.dumps(parsed),
    }
        )

        await ctx.send_message(ctx_input)


@executor(id="run_parallel_indexing")
async def run_parallel_indexing(
    ctx_input: dict,
    ctx: WorkflowContext[str]
):
    with tracer.start_as_current_span("indexing_agent_in_parallel") as span:  # ✅ ADDED
        classification = ctx_input.get("classification", {})

        splits = (
            classification.get("splitting", {}).get("splitted_blob_names", [])
            or classification.get("splitting", {}).get("segment_blob_names", [])
        )

        if not splits:
            ctx_input["indexing_results"] = []
            await ctx.yield_output(ctx_input)
            return
        workflow_id=ctx_input["workflow_id"]
        span.add_event(
            name="Indexing started",
            attributes={
            "workflow_id": workflow_id,  # ✅ UPDATED (dynamic)
            "input": json.dumps(classification),
    }
        )
        agent = build_indexing_agent()

        customer_name = classification.get("vendor", {}).get("selected_vendor")
        invoice_type = classification.get("classification", {}).get("Type_of_Invoice")
        market_name = classification.get("market")

        async def index_one(blob_name: str):
            prompt = f"""
Execute indexing.

Inputs:
- segment_blob_name: {blob_name}
- market_name: {market_name}
- customer_name: {customer_name}
- invoice_type: {invoice_type}
"""
            return await agent.run(prompt)

        results = await asyncio.gather(*(index_one(s) for s in splits))

        ctx_input["indexing_results"] = [
            json.loads(r.text) for r in results
        ]
        span.add_event(
            name="Indexing Completed",
            attributes={
            "workflow_id": workflow_id,  # ✅ UPDATED (dynamic)
            "output": json.dumps(ctx_input["indexing_results"]),
    }
        )

        await ctx.yield_output(ctx_input)


# Build Workflow
def create_workflow():
    return (
        WorkflowBuilder(start_executor=run_classification)
        .add_edge(run_classification, run_parallel_indexing)
        .build()
    )


# Run Workflow
async def main(workflow_id:str,pdf_blob_name: str, market: str):
    checkpoint_storage = FileCheckpointStorage(workflow_id)  # ✅ ADDED

    # pdf_blob_name="Albertsons companies-Sample.pdf"
    # market="USR"
    input_data = {
        "workflow_id": workflow_id,  # ✅ UPDATED (dynamic)
        "pdf_blob_name": pdf_blob_name,
        "market": market
    }
    with tracer.start_as_current_span("workflow_run") as span:
        span.add_event(
            name="Workflow Started",
            attributes={
            "workflow_id": workflow_id,  # ✅ UPDATED (dynamic)
            "pdf_blob_name": pdf_blob_name,
            "market": market
    }
        )

        events = await workflow.run(
            input_data,
            checkpoint_storage=checkpoint_storage  # ✅ ADDED
        )

        print("Output:", events.get_outputs())
        print("Final State:", events.get_final_state())
        span.add_event(
                name="Workflow Completed",
                attributes={
                "workflow_id": workflow_id,  # ✅ UPDATED (dynamic)
                "agent_output": json.dumps(events.get_outputs())
        }
        )
    


if __name__ == "__main__":
    workflow = create_workflow()
    
    workflow_id = f"wf-{uuid.uuid4()}"  # ✅ ADDED
    input_data = {
        "workflow_id": workflow_id,
        "pdf_blob_name": "Albertsons companies-Sample.pdf",
        #"pdf_blob_name":"7Elevn-07389746.pdf",
        #"pdf_blob_name":"ALW015351790-04.pdf",
        #"pdf_blob_name": "Albertson-one page.pdf",
        #"pdf_blob_name":"AHOLD DELHAIZE.pdf",
        #"pdf_blob_name":"Walmart-06850491.pdf",
        #"pdf_blob_name":"Walmart2-07389746.pdf",
        #"pdf_blob_name":"Racetrac_General Mills Inc CINV260216017634 2026-02-17.pdf",
        #"pdf_blob_name":"Ok Grocery.pdf",
        #"pdf_blob_name":"Weis-01433349.pdf",
        #"pdf_blob_name":"Bashas-Invoice.pdf",
        "market": "USR"
    }
    asyncio.run(main(workflow_id=input_data["workflow_id"],pdf_blob_name=input_data["pdf_blob_name"],market=input_data["market"]))