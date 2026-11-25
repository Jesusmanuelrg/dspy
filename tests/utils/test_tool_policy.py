from __future__ import annotations

import asyncio
import dataclasses
import pickle
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

import dspy
from dspy.adapters.types.tool import Tool, ToolAnnotations
from dspy.predict.flex import Flex
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.facade import SandboxSession
from dspy.primitives.local_interpreter import LocalInterpreter
from dspy.primitives.sandbox_guest import GuestValues, HostTool
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import DSPyError, ToolPolicyError
from dspy.utils.mcp import convert_mcp_tool
from dspy.utils.tool_policy import (
    Allow,
    Deny,
    Modify,
    ReadOnlyToolPolicy,
    ToolAllowlistPolicy,
    ToolApprovalPolicy,
    ToolCallBudgetPolicy,
    ToolCallContext,
    ToolPolicy,
    ToolPolicyDecision,
)


def _add(x: int, y: int = 1) -> int:
    """Add two integers."""
    return x + y


async def _aadd(x: int, y: int = 1) -> int:
    """Async add two integers."""
    await asyncio.sleep(0)
    return x + y


# ---------------------------------------------------------------------------
# ToolAnnotations & MCP preservation
# ---------------------------------------------------------------------------


def test_tool_annotations_immutable_and_preserves_function_tool_schema_and_pickle():
    ann = ToolAnnotations(title="Search", read_only=True, destructive=False, idempotent=True, open_world=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ann.read_only = False  # type: ignore[misc]

    plain = Tool(_add)
    annotated = Tool(_add, annotations=ann)

    assert plain.annotations is None
    assert annotated.annotations == ann
    assert annotated.as_function_tool() == plain.as_function_tool()

    restored = pickle.loads(pickle.dumps(annotated))
    assert restored.annotations == ann
    assert restored(x=2, y=3) == 5


@pytest.mark.parametrize(
    "raw_annotations",
    [
        pytest.param(
            SimpleNamespace(
                title="Lookup",
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            id="camel-object",
        ),
        pytest.param(
            SimpleNamespace(
                title="Lookup",
                read_only_hint=True,
                destructive_hint=False,
                idempotent_hint=True,
                open_world_hint=False,
            ),
            id="snake-object",
        ),
        pytest.param(
            {
                "title": "Lookup",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
            id="camel-dict",
        ),
        pytest.param(
            {
                "title": "Lookup",
                "read_only_hint": True,
                "destructive_hint": False,
                "idempotent_hint": True,
                "open_world_hint": False,
            },
            id="snake-dict",
        ),
    ],
)
def test_convert_mcp_tool_preserves_annotations(raw_annotations):
    mcp_tool = SimpleNamespace(
        name="lookup",
        description="Look up a record",
        inputSchema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        annotations=raw_annotations,
    )
    tool = convert_mcp_tool(SimpleNamespace(), mcp_tool)
    assert tool.annotations == ToolAnnotations(
        title="Lookup",
        read_only=True,
        destructive=False,
        idempotent=True,
        open_world=False,
    )


def test_convert_mcp_tool_without_annotations_sets_none():
    mcp_tool = SimpleNamespace(name="ping", description="Ping", input_schema={})
    tool = convert_mcp_tool(SimpleNamespace(), mcp_tool)
    assert tool.annotations is None


def test_tool_annotations_sandbox_wire_roundtrip_and_backward_compat():
    ann = ToolAnnotations(title="Reader", read_only=True, destructive=False)
    tool = Tool(_add, name="reader", annotations=ann)
    session = SandboxSession({"reader": tool})
    guest = GuestValues()

    spec = session._spec("test")["tools"]["reader"]
    assert spec["annotations"] == dataclasses.asdict(ann)

    host_tool = HostTool(spec, session._serve_tool, guest)
    assert host_tool.annotations == ann

    # Wire encode/decode preserves annotations.
    wire_node = session.encode(tool)
    decoded = guest.decode(wire_node)
    assert decoded.annotations == ann

    # Backward compatibility: legacy wire/spec dicts without "annotations" still decode cleanly.
    legacy_spec = {k: v for k, v in spec.items() if k != "annotations"}
    legacy_host_tool = HostTool(legacy_spec, session._serve_tool, guest)
    assert legacy_host_tool.annotations is None

    legacy_node = {k: v for k, v in wire_node.items() if k != "annotations"}
    assert guest.decode(legacy_node).annotations is None


# ---------------------------------------------------------------------------
# Tool.__call__ / Tool.acall mediation, composition, fail-closed semantics
# ---------------------------------------------------------------------------


def test_tool_policy_allow_deny_modify_after_call_sync():
    called = []

    def work(x: int, y: int = 1) -> int:
        called.append((x, y))
        return x + y

    tool = Tool(work)

    class ModifyAndTag(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            assert ctx.tool_name == "work"
            assert ctx.tool is tool
            assert ctx.caller == "tool"
            return Modify({"x": ctx.args["x"] * 10, "y": ctx.args.get("y", 1)})

        def after_call(self, ctx: ToolCallContext, result: Any) -> Any:
            return {"sum": result, "x_used": ctx.args["x"]}

    with dspy.context(tool_policy=ModifyAndTag()):
        assert tool(x=3, y=2) == {"sum": 32, "x_used": 30}
    assert called == [(30, 2)]

    # Deny blocks execution and raises ToolPolicyError (both DSPyError & PermissionError).
    called.clear()

    class AlwaysDeny(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            return Deny("forbidden")

    with dspy.context(tool_policy=AlwaysDeny()):
        with pytest.raises(ToolPolicyError, match="forbidden") as exc_info:
            tool(x=1)
    assert isinstance(exc_info.value, DSPyError)
    assert isinstance(exc_info.value, PermissionError)
    assert exc_info.value.tool_name == "work"
    assert exc_info.value.reason == "forbidden"
    assert called == []


@pytest.mark.asyncio
async def test_tool_policy_allow_deny_modify_after_call_async():
    called = []

    async def awork(x: int, y: int = 1) -> int:
        called.append((x, y))
        await asyncio.sleep(0)
        return x + y

    tool = Tool(awork)

    class AsyncRewritePolicy(ToolPolicy):
        async def abefore_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            await asyncio.sleep(0)
            return ToolPolicyDecision.modify({"x": ctx.args["x"] + 5, "y": 2})

        async def aafter_call(self, ctx: ToolCallContext, result: Any) -> Any:
            await asyncio.sleep(0)
            return result * 100

    with dspy.context(tool_policy=AsyncRewritePolicy()):
        assert await tool.acall(x=3) == 1000
    assert called == [(8, 2)]

    called.clear()
    with dspy.context(tool_policy=ToolAllowlistPolicy(denied=["awork"])):
        with pytest.raises(ToolPolicyError, match="denied"):
            await tool.acall(x=3)
    assert called == []


def test_tool_policy_modify_revalidates_args():
    called = []

    def work(x: int) -> int:
        called.append(x)
        return x

    tool = Tool(work)

    class BadModify(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            return Modify({"x": "not-an-int"})

    with dspy.context(tool_policy=BadModify()):
        with pytest.raises(ValueError, match="Arg x is invalid"):
            tool(x=1)
    assert called == []


@pytest.mark.asyncio
async def test_tool_policy_modify_revalidates_args_async():
    called = []

    async def awork(x: int) -> int:
        called.append(x)
        return x

    tool = Tool(awork)

    class BadModify(ToolPolicy):
        async def abefore_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            return Modify({"unknown_arg": 1})

    with dspy.context(tool_policy=BadModify()):
        with pytest.raises(ValueError, match="not in the tool's args"):
            await tool.acall(x=1)
    assert called == []


def test_tool_policy_composition_order_short_circuit_and_chaining():
    events = []

    class Step1(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            events.append(("b1", ctx.args["x"]))
            return Modify({"x": ctx.args["x"] + 1})

        def after_call(self, ctx: ToolCallContext, result: Any) -> Any:
            events.append(("a1", result))
            return result + 10

    class Step2(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            events.append(("b2", ctx.args["x"]))
            return Modify({"x": ctx.args["x"] * 2})

        def after_call(self, ctx: ToolCallContext, result: Any) -> Any:
            events.append(("a2", result))
            return result * 3

    tool = Tool(lambda x: x, name="id", args={"x": {"type": "integer"}}, arg_types={"x": int})
    with dspy.context(tool_policy=[Step1(), Step2()]):
        assert tool(x=4) == ((4 + 1) * 2 + 10) * 3
    assert events == [("b1", 4), ("b2", 5), ("a1", 10), ("a2", 20)]

    # Short-circuit on first Deny: subsequent budget policy must NOT be called.
    budget = ToolCallBudgetPolicy(max_calls=1)
    with dspy.context(tool_policy=[ToolAllowlistPolicy(allowed=["other"]), budget]):
        with pytest.raises(ToolPolicyError):
            tool(x=1)
    assert budget.total_calls == 0


@pytest.mark.parametrize("bad_decision", [None, True, False, "allow", 42, object()])
def test_tool_policy_fail_closed_on_unrecognized_decision(bad_decision):
    called = []

    def work(x: int) -> int:
        called.append(x)
        return x

    class BrokenPolicy(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> Any:
            return bad_decision

    with dspy.context(tool_policy=BrokenPolicy()):
        with pytest.raises(ToolPolicyError, match="unsupported decision"):
            Tool(work)(x=1)
    assert called == []


def test_tool_policy_propagates_policy_exceptions_and_rejects_invalid_config():
    called = []

    def work(x: int) -> int:
        called.append(x)
        return x

    class CrashPolicy(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            raise RuntimeError("policy database unreachable")

    with dspy.context(tool_policy=CrashPolicy()):
        with pytest.raises(RuntimeError, match="policy database unreachable"):
            Tool(work)(x=1)
    assert called == []

    for bad_policy in ("allow", b"allow", 123, [object()]):
        with dspy.context(tool_policy=bad_policy):
            with pytest.raises(ToolPolicyError, match="Invalid tool_policy"):
                Tool(work)(x=1)
    assert called == []


def test_tool_call_execute_enforces_policy():
    called = []

    def secret_op(x: int) -> int:
        called.append(x)
        return x + 1

    tool = Tool(secret_op)
    tc = dspy.ToolCalls.ToolCall(name="secret_op", args={"x": 5})

    with dspy.context(tool_policy=ToolAllowlistPolicy(denied=["secret_op"])):
        with pytest.raises(ToolPolicyError):
            tc.execute(functions=[tool])
        with pytest.raises(ToolPolicyError):
            tc.execute(functions={"secret_op": secret_op})
        with pytest.raises(ToolPolicyError):
            tc.execute()
    assert called == []

    budget = ToolCallBudgetPolicy(max_calls=1)
    with dspy.context(tool_policy=budget):
        assert tc.execute(functions=[tool]) == 6
    assert budget.total_calls == 1


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


def test_tool_allowlist_policy():
    t_read = Tool(lambda: "r", name="read")
    t_write = Tool(lambda: "w", name="write")

    # Deny takes precedence over allow.
    policy = ToolAllowlistPolicy(allowed=["read", "write"], denied=["write"])
    with dspy.context(tool_policy=policy):
        assert t_read() == "r"
        with pytest.raises(ToolPolicyError):
            t_write()

    # Empty allowlist denies everything.
    with dspy.context(tool_policy=ToolAllowlistPolicy(allowed=[])):
        with pytest.raises(ToolPolicyError):
            t_read()

    # Bare string rejected to avoid character-splitting footgun.
    with pytest.raises(TypeError):
        ToolAllowlistPolicy(allowed="read")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ToolAllowlistPolicy(denied="write")  # type: ignore[arg-type]


def test_read_only_tool_policy():
    t_ro = Tool(lambda: "ok", name="ro", annotations=ToolAnnotations(read_only=True))
    t_rw = Tool(lambda: "ok", name="rw", annotations=ToolAnnotations(read_only=False))
    t_destr = Tool(lambda: "ok", name="destr", annotations=ToolAnnotations(read_only=True, destructive=True))
    t_unannotated = Tool(lambda: "ok", name="plain")
    t_unknown = Tool(lambda: "ok", name="unknown", annotations=ToolAnnotations(title="Unknown"))

    # Default require_annotation=True fails closed.
    with dspy.context(tool_policy=ReadOnlyToolPolicy(require_annotation=True)):
        assert t_ro() == "ok"
        for blocked in (t_rw, t_destr, t_unannotated, t_unknown):
            with pytest.raises(ToolPolicyError):
                blocked()

    # require_annotation=False permits unannotated/unspecified, still blocks destructive & read_only=False.
    with dspy.context(tool_policy=ReadOnlyToolPolicy(require_annotation=False)):
        assert t_ro() == "ok"
        assert t_unannotated() == "ok"
        assert t_unknown() == "ok"
        for blocked in (t_rw, t_destr):
            with pytest.raises(ToolPolicyError):
                blocked()


def test_tool_call_budget_policy_limits_and_thread_safety():
    t_a = Tool(lambda: "a", name="a")
    t_b = Tool(lambda: "b", name="b")

    budget = ToolCallBudgetPolicy(max_calls=3, per_tool={"a": 2})
    with dspy.context(tool_policy=budget):
        assert t_a() == "a"
        assert t_a() == "a"
        with pytest.raises(ToolPolicyError, match="budget exceeded"):
            t_a()
        # Denied call did not consume total budget.
        assert budget.total_calls == 2
        assert t_b() == "b"
        with pytest.raises(ToolPolicyError, match="budget exceeded"):
            t_b()
    assert budget.total_calls == 3
    assert budget.tool_calls == {"a": 2, "b": 1}

    budget.reset()
    assert budget.total_calls == 0

    # Concurrent stress test: exactly 10 calls succeed out of 40 concurrent attempts.
    shared_budget = ToolCallBudgetPolicy(max_calls=10)
    counter_lock = threading.Lock()
    succeeded = 0
    denied = 0

    def worker():
        nonlocal succeeded, denied
        with dspy.context(tool_policy=shared_budget):
            try:
                t_a()
                with counter_lock:
                    succeeded += 1
            except ToolPolicyError:
                with counter_lock:
                    denied += 1

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: worker(), range(40)))

    assert succeeded == 10
    assert denied == 30
    assert shared_budget.total_calls == 10


@pytest.mark.asyncio
async def test_tool_approval_policy_sync_async_and_only_if():
    prompts = []

    def sync_approver(ctx: ToolCallContext) -> bool:
        prompts.append((ctx.tool_name, ctx.args))
        return ctx.args.get("ok", False)

    t_safe = Tool(
        lambda ok=False: "safe",
        name="safe",
        annotations=ToolAnnotations(read_only=True),
    )
    t_risky = Tool(
        lambda ok=False: "risky",
        name="risky",
        annotations=ToolAnnotations(read_only=False, destructive=True),
    )

    policy = ToolApprovalPolicy(
        approver=sync_approver,
        only_if=lambda ctx: ctx.annotations is None or ctx.annotations.read_only is not True,
    )
    with dspy.context(tool_policy=policy):
        assert t_safe() == "safe"
        assert prompts == []  # skipped approver for read-only tool
        with pytest.raises(ToolPolicyError, match="not approved"):
            t_risky(ok=False)
        assert t_risky(ok=True) == "risky"
    assert len(prompts) == 2

    # Async approver works in acall, fails closed in sync __call__.
    async def async_approver(ctx: ToolCallContext) -> ToolPolicyDecision:
        await asyncio.sleep(0)
        return Allow() if ctx.args.get("ok") else Deny("human clicked no")

    async_policy = ToolApprovalPolicy(approver=async_approver)
    with dspy.context(tool_policy=async_policy):
        assert await t_risky.acall(ok=True) == "risky"
        with pytest.raises(ToolPolicyError, match="human clicked no"):
            await t_risky.acall(ok=False)
        with pytest.raises(ToolPolicyError, match="synchronously"):
            t_risky(ok=True)


# ---------------------------------------------------------------------------
# Context isolation across threads and async tasks
# ---------------------------------------------------------------------------


def test_tool_policy_context_isolation_across_threads():
    tool = Tool(lambda: "ok", name="op")
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}

    def run_thread(label: str, policy: ToolPolicy | None):
        with dspy.context(tool_policy=policy):
            barrier.wait()
            try:
                results[label] = tool()
            except ToolPolicyError:
                results[label] = "denied"

    t1 = threading.Thread(target=run_thread, args=("allowed", ToolAllowlistPolicy(allowed=["op"])))
    t2 = threading.Thread(target=run_thread, args=("denied", ToolAllowlistPolicy(denied=["op"])))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results == {"allowed": "ok", "denied": "denied"}
    assert dspy.settings.tool_policy is None


@pytest.mark.asyncio
async def test_tool_policy_context_isolation_across_async_tasks():
    tool = Tool(_aadd, name="aadd")
    gate = asyncio.Event()

    async def task_allowed():
        with dspy.context(tool_policy=ToolAllowlistPolicy(allowed=["aadd"])):
            await gate.wait()
            return await tool.acall(x=2, y=3)

    async def task_denied():
        with dspy.context(tool_policy=ToolAllowlistPolicy(denied=["aadd"])):
            await gate.wait()
            try:
                return await tool.acall(x=2, y=3)
            except ToolPolicyError:
                return "denied"

    t1 = asyncio.create_task(task_allowed())
    t2 = asyncio.create_task(task_denied())
    await asyncio.sleep(0)
    gate.set()
    assert await asyncio.gather(t1, t2) == [5, "denied"]


# ---------------------------------------------------------------------------
# ReAct: sync & async recovery on denial + submit exemption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asynchronous", [False, True])
def test_react_policy_denial_recovery_and_submit_exemption(asynchronous):
    executed = []
    seen_callers = []

    def delete_all(target: str) -> str:
        executed.append(("delete_all", target))
        return f"deleted {target}"

    def lookup(query: str) -> str:
        executed.append(("lookup", query))
        return f"result:{query}"

    class CallerRecorder(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            seen_callers.append((ctx.tool_name, ctx.caller))
            return Allow()

    tools = [
        Tool(delete_all, annotations=ToolAnnotations(read_only=False, destructive=True)),
        Tool(lookup, annotations=ToolAnnotations(read_only=True)),
    ]

    lm = DummyLM(
        [
            {
                "next_thought": "Try delete_all first.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "delete_all", "args": {"target": "db"}}]),
            },
            {
                "next_thought": "Blocked; fall back to lookup.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "status"}}]),
            },
            {
                "next_thought": "Submit final answer.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "result:status"}}]),
            },
        ]
    )

    budget = ToolCallBudgetPolicy(max_calls=1)
    policies = [
        CallerRecorder(),
        ToolAllowlistPolicy(allowed=["lookup"]),
        ReadOnlyToolPolicy(require_annotation=True),
        budget,
    ]
    agent = dspy.ReAct("question -> answer", tools=tools)

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), tool_policy=policies):
        pred = asyncio.run(agent.acall(question="check")) if asynchronous else agent(question="check")

    assert pred.answer == "result:status"
    assert pred.termination_reason == "submit"
    assert executed == [("lookup", "status")]
    # Budget reached 1 on `lookup`, yet internal `submit` still succeeded without being counted or blocked.
    assert budget.total_calls == 1
    assert seen_callers == [("delete_all", "react"), ("lookup", "react")]
    first_result = pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0]
    assert first_result.is_error is True
    assert "ToolPolicyError" in str(first_result.value)


# ---------------------------------------------------------------------------
# RLM & Flex (LocalInterpreter): exactly-once mediation for Tool & plain functions
# ---------------------------------------------------------------------------


def test_rlm_local_interpreter_enforces_policy_once_and_recovers():
    executed = []

    def secret_write(val: str) -> str:
        """Write a value."""
        executed.append(("secret_write", val))
        return f"wrote:{val}"

    def safe_read(key: str) -> str:
        """Read a value."""
        executed.append(("safe_read", key))
        return f"val:{key}"

    safe_tool = Tool(safe_read, annotations=ToolAnnotations(read_only=True))
    # Pass one plain function (`secret_write`) and one `dspy.Tool` (`safe_tool`),
    # and call both directly AND after `import dspy` (via HostTool / __dspy_tool_call__).
    lm = DummyLM(
        [
            {
                "plan": "Try blocked plain tool first.",
                "code": "print(secret_write('x'))",
            },
            {
                "plan": "Call allowed tool directly, then import dspy and call again via HostTool.",
                "code": ("r1 = safe_read('k1')\nimport dspy\nr2 = safe_read('k2')\nSUBMIT(answer=f'{r1}|{r2}')\n"),
            },
        ]
    )

    budget = ToolCallBudgetPolicy(max_calls=2)
    policies = [ReadOnlyToolPolicy(require_annotation=True), budget]

    rlm = dspy.RLM(
        "question -> answer",
        tools=[secret_write, safe_tool],
        interpreter_factory=LocalInterpreter,
        max_iters=3,
    )
    with dspy.context(lm=lm, tool_policy=policies):
        pred = rlm(question="read k1 and k2")

    assert pred.answer == "val:k1|val:k2"
    assert executed == [("safe_read", "k1"), ("safe_read", "k2")]
    # Exactly 2 tool calls counted — no double-counting across LocalInterpreter / HostTool / __dspy_tool_call__,
    # and internal bridge tools (__dspy_lm__, __dspy_trace__, __dspy_tool_call__, SUBMIT) were not counted or blocked.
    assert budget.total_calls == 2
    assert "ToolPolicyError" in pred.trajectory[0]["output"]


def test_flex_local_interpreter_mediates_plain_functions_and_tools_exactly_once():
    executed = []

    def plain_shout(text: str) -> str:
        """Uppercase text."""
        executed.append(("plain_shout", text))
        return text.upper()

    def wrapped_echo(text: str) -> str:
        """Echo text."""
        executed.append(("wrapped_echo", text))
        return f"echo:{text}"

    wrapped_tool = Tool(wrapped_echo, annotations=ToolAnnotations(read_only=True))

    src = textwrap.dedent(
        """
        class PipeModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q):
                return dspy.Prediction(a=plain_shout(q) + "|" + wrapped_echo(q))
        """
    ).strip()

    class PrefixAndRedactor(ToolPolicy):
        def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
            if ctx.tool_name == "plain_shout":
                return Modify({"text": f"mod:{ctx.args['text']}"})
            return Allow()

        def after_call(self, ctx: ToolCallContext, result: Any) -> Any:
            return f"[{result}]"

    program = Flex("q -> a", tools=[plain_shout, wrapped_tool], interpreter_factory=LocalInterpreter)
    program._bind_code(src)

    # Both plain function and dspy.Tool are mediated exactly once.
    budget = ToolCallBudgetPolicy(max_calls=2)
    with dspy.context(tool_policy=[PrefixAndRedactor(), budget]):
        pred = program(q="hi")

    assert pred.a == "[MOD:HI]|[echo:hi]"
    assert executed == [("plain_shout", "mod:hi"), ("wrapped_echo", "hi")]
    assert budget.total_calls == 2

    # Denying the plain function blocks execution (no bypass!).
    executed.clear()
    with dspy.context(tool_policy=ToolAllowlistPolicy(allowed=["wrapped_echo"])):
        with pytest.raises(CodeInterpreterError, match="ToolPolicyError"):
            program(q="hi")
    assert executed == []


def test_flex_nested_rlm_mediates_plain_function_exactly_once():
    executed = []

    def secret_lookup(key: str) -> str:
        """Look up key."""
        executed.append(key)
        return f"found:{key}"

    lm = DummyLM(
        [
            {"plan": "Call secret_lookup.", "code": "SUBMIT(a=secret_lookup('abc'))"},
            {"plan": "Call secret_lookup.", "code": "SUBMIT(a=secret_lookup('def'))"},
        ]
    )
    program = Flex("q -> a", tools=[secret_lookup], interpreter_factory=LocalInterpreter)

    budget = ToolCallBudgetPolicy(max_calls=1)
    with dspy.context(lm=lm, tool_policy=budget):
        pred = program(q="find abc")
    assert pred.a == "found:abc"
    assert executed == ["abc"]
    assert budget.total_calls == 1

    # With allowlist excluding secret_lookup, nested RLM cannot execute it.
    executed.clear()
    lm2 = DummyLM(
        [
            {"plan": "Try secret_lookup.", "code": "print(secret_lookup('blocked'))"},
            {"plan": "Fallback.", "code": "SUBMIT(a='blocked')"},
        ]
    )
    with dspy.context(lm=lm2, tool_policy=ToolAllowlistPolicy(allowed=[])):
        pred2 = program(q="find blocked")
    assert pred2.a == "blocked"
    assert executed == []


def test_sandbox_session_serve_tool_mediates_plain_callable_once():
    calls = []

    def plain_fn(x: int) -> int:
        calls.append(x)
        return x * 2

    session = SandboxSession({"plain_fn": plain_fn})
    budget = ToolCallBudgetPolicy(max_calls=1)

    with dspy.context(tool_policy=budget):
        assert session._serve_tool("plain_fn", {"x": 7}) == 14
        with pytest.raises(ToolPolicyError, match="budget exceeded"):
            session._serve_tool("plain_fn", {"x": 8})

    assert calls == [7]
    assert budget.total_calls == 1
