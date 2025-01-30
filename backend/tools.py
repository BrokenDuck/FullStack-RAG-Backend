from cohere import ToolV2, ToolV2Function

query_generation_tool = ToolV2(
    function=ToolV2Function(
        name="database_search",
        parameters={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "a list of queries for similarity search in a database.",
                }
            },
            "required": ["queries"],
        },
    )
)
