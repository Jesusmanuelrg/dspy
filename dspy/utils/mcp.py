from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from dspy.adapters.types.tool import Tool, ToolAnnotations, _MCPToolClient, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    import mcp


def _get_field(obj: Any, snake_name: str, camel_name: str, default: Any = None) -> Any:
    """Read a field whose Python name changed between mcp SDK v1 and v2."""
    names = (snake_name,) if snake_name == camel_name else (snake_name, camel_name)
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _extract_mcp_annotations(tool: Any) -> ToolAnnotations | None:
    raw = _get_field(tool, "annotations", "annotations", default=None)
    if raw is None:
        return None
    if isinstance(raw, ToolAnnotations):
        return raw
    return ToolAnnotations(
        title=_get_field(raw, "title", "title", default=None),
        read_only=_get_field(raw, "read_only_hint", "readOnlyHint", default=None),
        destructive=_get_field(raw, "destructive_hint", "destructiveHint", default=None),
        idempotent=_get_field(raw, "idempotent_hint", "idempotentHint", default=None),
        open_world=_get_field(raw, "open_world_hint", "openWorldHint", default=None),
    )


def _field_is_set(obj: Any, snake_name: str, camel_name: str) -> bool:
    """Check whether an optional field was explicitly set, including to null."""
    fields_set = getattr(obj, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(obj, "__fields_set__", None)
    if fields_set is not None:
        return snake_name in fields_set or camel_name in fields_set
    return hasattr(obj, snake_name) or hasattr(obj, camel_name)


def _convert_mcp_tool_result(
    call_tool_result: "mcp.types.CallToolResult",
    result_mode: Literal["text", "structured"] = "text",
) -> Any:
    from mcp.types import TextContent

    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if _get_field(call_tool_result, "is_error", "isError", default=False):
        raise RuntimeError(f"Failed to call a MCP tool: {tool_content}")

    if result_mode == "structured" and _field_is_set(call_tool_result, "structured_content", "structuredContent"):
        return _get_field(call_tool_result, "structured_content", "structuredContent")

    return tool_content or non_text_contents


def convert_mcp_tool(
    session: _MCPToolClient,
    tool: "mcp.types.Tool",
    *,
    result_mode: Literal["text", "structured"] = "text",
) -> Tool:
    """Build a DSPy tool from an MCP tool.

    Both mcp SDK v1's ``ClientSession`` and v2's high-level ``Client`` satisfy
    the client interface used by this bridge.

    Args:
        session: An MCP client or session with an async ``call_tool`` method.
        tool: The MCP tool to convert.
        result_mode: ``"text"`` preserves DSPy's existing text/non-text
            conversion. ``"structured"`` returns MCP structured content
            exactly when present, with the existing conversion as fallback.

    Returns:
        A dspy Tool object.
    """
    if result_mode not in ("text", "structured"):
        raise ValueError(f"Unsupported MCP result mode: {result_mode!r}")

    input_schema = _get_field(tool, "input_schema", "inputSchema", default={})
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(input_schema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result, result_mode=result_mode)

    annotations = _extract_mcp_annotations(tool)

    return Tool(
        func=func,
        name=tool.name,
        desc=tool.description,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc,
        annotations=annotations,
    )
