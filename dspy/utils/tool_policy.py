from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import inspect
import threading
from collections.abc import Callable, Collection, Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dspy.dsp.utils.settings import settings
from dspy.utils.exceptions import ToolPolicyError

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool, ToolAnnotations

_TOOL_CALLER: contextvars.ContextVar[str] = contextvars.ContextVar("dspy_tool_caller", default="tool")


def get_tool_caller() -> str:
    return _TOOL_CALLER.get()


@contextlib.contextmanager
def tool_caller_context(caller: str) -> Generator[None, None, None]:
    token = _TOOL_CALLER.set(caller)
    try:
        yield
    finally:
        _TOOL_CALLER.reset(token)


@dataclass(frozen=True)
class ToolCallContext:
    """Context describing a single tool invocation for policy evaluation."""

    tool_name: str
    args: dict[str, Any]
    tool: Tool | None = None
    annotations: ToolAnnotations | None = None
    caller: str = "tool"


class ToolPolicyDecision:
    """Base class for decisions returned by ``ToolPolicy.before_call``."""

    @classmethod
    def allow(cls) -> Allow:
        return Allow()

    @classmethod
    def deny(cls, reason: str = "") -> Deny:
        return Deny(reason=reason)

    @classmethod
    def modify(cls, args: dict[str, Any]) -> Modify:
        return Modify(args=args)


@dataclass(frozen=True)
class Allow(ToolPolicyDecision):
    """Allow the tool call to proceed with its current arguments."""


@dataclass(frozen=True)
class Deny(ToolPolicyDecision):
    """Block the tool call and raise ``ToolPolicyError``."""

    reason: str = ""


@dataclass(frozen=True)
class Modify(ToolPolicyDecision):
    """Allow the tool call with replacement keyword arguments."""

    args: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.args, dict) or not all(isinstance(k, str) for k in self.args):
            raise ToolPolicyError(f"Modify decision requires a dict with string keys, got {type(self.args).__name__}.")


class ToolPolicy:
    """Base class for runtime tool execution policies.

    Subclasses override ``before_call`` / ``abefore_call`` to allow, deny, or
    modify tool invocations before execution, and optionally ``after_call`` /
    ``aafter_call`` to inspect or transform tool results.
    """

    def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
        return Allow()

    async def abefore_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
        return self.before_call(ctx)

    def after_call(self, ctx: ToolCallContext, result: Any) -> Any:
        return result

    async def aafter_call(self, ctx: ToolCallContext, result: Any) -> Any:
        return self.after_call(ctx, result)


class ToolAllowlistPolicy(ToolPolicy):
    """Restrict tool calls by explicit allowlist and/or denylist.

    ``denied`` always takes precedence over ``allowed``. When ``allowed`` is
    provided (including an empty collection), only listed tool names may run.
    """

    def __init__(
        self,
        allowed: Collection[str] | None = None,
        denied: Collection[str] | None = None,
    ) -> None:
        if isinstance(allowed, (str, bytes, bytearray)):
            raise TypeError("allowed must be a collection of tool names, not a string or bytes.")
        if isinstance(denied, (str, bytes, bytearray)):
            raise TypeError("denied must be a collection of tool names, not a string or bytes.")
        self.allowed: frozenset[str] | None = frozenset(allowed) if allowed is not None else None
        self.denied: frozenset[str] = frozenset(denied) if denied is not None else frozenset()

    def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
        if ctx.tool_name in self.denied:
            return Deny(f"Tool '{ctx.tool_name}' is denied by policy.")
        if self.allowed is not None and ctx.tool_name not in self.allowed:
            return Deny(f"Tool '{ctx.tool_name}' is not in the allowed tool list.")
        return Allow()


class ReadOnlyToolPolicy(ToolPolicy):
    """Allow only read-only tools based on ``ToolAnnotations``.

    When ``require_annotation=True`` (the default), calls fail closed unless the
    tool carries ``annotations`` with ``read_only=True`` and ``destructive`` not
    ``True``.
    """

    def __init__(self, require_annotation: bool = True) -> None:
        self.require_annotation = require_annotation

    def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
        annotations = ctx.annotations
        if annotations is None and ctx.tool is not None:
            annotations = getattr(ctx.tool, "annotations", None)

        if annotations is None:
            if self.require_annotation:
                return Deny(f"Tool '{ctx.tool_name}' has no ToolAnnotations; read_only=True is required.")
            return Allow()

        if annotations.destructive is True:
            return Deny(f"Tool '{ctx.tool_name}' is marked destructive.")
        if annotations.read_only is False:
            return Deny(f"Tool '{ctx.tool_name}' is not read-only.")
        if annotations.read_only is True:
            return Allow()
        if self.require_annotation:
            return Deny(f"Tool '{ctx.tool_name}' lacks read_only=True annotation.")
        return Allow()


class ToolCallBudgetPolicy(ToolPolicy):
    """Thread-safe cap on total and/or per-tool invocations."""

    def __init__(
        self,
        max_calls: int | None = None,
        per_tool: dict[str, int] | None = None,
    ) -> None:
        if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0):
            raise ValueError(f"max_calls must be a non-negative int or None, got {max_calls!r}")
        validated_per_tool: dict[str, int] = {}
        if per_tool is not None:
            for name, limit in per_tool.items():
                if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                    raise ValueError(f"per_tool limit for {name!r} must be a non-negative int, got {limit!r}")
                validated_per_tool[name] = limit
        self.max_calls = max_calls
        self.per_tool = validated_per_tool
        self._lock = threading.Lock()
        self._total_calls = 0
        self._tool_calls: dict[str, int] = {}

    @property
    def total_calls(self) -> int:
        with self._lock:
            return self._total_calls

    @property
    def tool_calls(self) -> dict[str, int]:
        with self._lock:
            return dict(self._tool_calls)

    def reset(self) -> None:
        with self._lock:
            self._total_calls = 0
            self._tool_calls.clear()

    def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
        with self._lock:
            if self.max_calls is not None and self._total_calls >= self.max_calls:
                return Deny(f"Tool call budget exceeded ({self.max_calls} total calls).")
            tool_limit = self.per_tool.get(ctx.tool_name)
            current = self._tool_calls.get(ctx.tool_name, 0)
            if tool_limit is not None and current >= tool_limit:
                return Deny(f"Tool '{ctx.tool_name}' call budget exceeded ({tool_limit} calls).")
            self._total_calls += 1
            self._tool_calls[ctx.tool_name] = current + 1
            return Allow()


class ToolApprovalPolicy(ToolPolicy):
    """Human-in-the-loop approval policy supporting sync and async approvers.

    Args:
        approver: Callable taking ``ToolCallContext`` and returning ``bool`` or
            a ``ToolPolicyDecision``. May be an async function when invoked via
            ``acall``.
        only_if: Optional predicate taking ``ToolCallContext``. When it returns
            a falsy value, the call proceeds without invoking ``approver``.
    """

    def __init__(
        self,
        approver: Callable[[ToolCallContext], Any],
        only_if: Callable[[ToolCallContext], Any] | None = None,
    ) -> None:
        if not callable(approver):
            raise TypeError(f"approver must be callable, got {type(approver).__name__}")
        if only_if is not None and not callable(only_if):
            raise TypeError(f"only_if must be callable or None, got {type(only_if).__name__}")
        self.approver = approver
        self.only_if = only_if

    def _interpret(self, ctx: ToolCallContext, outcome: Any) -> ToolPolicyDecision:
        if isinstance(outcome, bool):
            return Allow() if outcome else Deny(f"Tool '{ctx.tool_name}' was not approved.")
        if isinstance(outcome, ToolPolicyDecision):
            return outcome
        raise ToolPolicyError(
            f"ToolApprovalPolicy approver returned unsupported value {outcome!r}; expected bool or ToolPolicyDecision.",
            tool_name=ctx.tool_name,
            policy=self,
        )

    def before_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
        if self.only_if is not None:
            needs_approval = self.only_if(ctx)
            if inspect.isawaitable(needs_approval):
                if inspect.iscoroutine(needs_approval):
                    needs_approval.close()
                raise ToolPolicyError(
                    "Async `only_if` predicate cannot be evaluated synchronously; use `acall`.",
                    tool_name=ctx.tool_name,
                    policy=self,
                )
            if not needs_approval:
                return Allow()

        outcome = self.approver(ctx)
        if inspect.isawaitable(outcome):
            if inspect.iscoroutine(outcome):
                outcome.close()
            raise ToolPolicyError(
                "Async approver cannot be evaluated synchronously; use `acall`.",
                tool_name=ctx.tool_name,
                policy=self,
            )
        return self._interpret(ctx, outcome)

    async def abefore_call(self, ctx: ToolCallContext) -> ToolPolicyDecision:
        if self.only_if is not None:
            needs_approval = self.only_if(ctx)
            if inspect.isawaitable(needs_approval):
                needs_approval = await needs_approval
            if not needs_approval:
                return Allow()

        outcome = self.approver(ctx)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return self._interpret(ctx, outcome)


def _validate_policy(policy: Any) -> None:
    if isinstance(policy, ToolPolicy):
        return
    if any(
        callable(getattr(policy, attr, None)) for attr in ("before_call", "abefore_call", "after_call", "aafter_call")
    ):
        return
    raise ToolPolicyError(
        f"Invalid tool_policy: expected ToolPolicy or sequence of ToolPolicy, got {type(policy).__name__}.",
        policy=policy,
    )


def resolve_tool_policies(policy: Any) -> tuple[Any, ...]:
    """Normalize a single policy or sequence of policies; fail closed on invalid entries."""
    if policy is None:
        return ()
    if isinstance(policy, ToolPolicy):
        return (policy,)
    if isinstance(policy, (str, bytes, bytearray)):
        raise ToolPolicyError(
            f"Invalid tool_policy: expected ToolPolicy or sequence of ToolPolicy, got {type(policy).__name__}.",
            policy=policy,
        )
    if isinstance(policy, Sequence):
        resolved = tuple(policy)
        for item in resolved:
            _validate_policy(item)
        return resolved
    _validate_policy(policy)
    return (policy,)


def get_active_tool_policies() -> tuple[Any, ...]:
    return resolve_tool_policies(settings.get("tool_policy", None))


def _apply_decision(policy: Any, ctx: ToolCallContext, decision: Any) -> tuple[ToolCallContext, bool]:
    if isinstance(decision, Allow):
        return ctx, False
    if isinstance(decision, Modify):
        return dataclasses.replace(ctx, args=dict(decision.args)), True
    if isinstance(decision, Deny):
        detail = decision.reason or "denied by policy"
        raise ToolPolicyError(
            f"Tool '{ctx.tool_name}' blocked by {type(policy).__name__}: {detail}",
            tool_name=ctx.tool_name,
            policy=policy,
            reason=decision.reason,
        )
    raise ToolPolicyError(
        f"Tool policy {type(policy).__name__} returned unsupported decision {decision!r}; "
        "expected Allow, Deny, or Modify.",
        tool_name=ctx.tool_name,
        policy=policy,
    )


def evaluate_before_call(policies: Sequence[Any], ctx: ToolCallContext) -> tuple[ToolCallContext, bool]:
    current_ctx = ctx
    modified = False
    for policy in policies:
        before = getattr(policy, "before_call", None)
        if before is None:
            if callable(getattr(policy, "abefore_call", None)):
                raise ToolPolicyError(
                    f"Async-only policy {type(policy).__name__} cannot be evaluated synchronously; use `acall`.",
                    tool_name=current_ctx.tool_name,
                    policy=policy,
                )
            continue
        decision = before(current_ctx)
        if inspect.isawaitable(decision):
            if inspect.iscoroutine(decision):
                decision.close()
            raise ToolPolicyError(
                f"Policy {type(policy).__name__}.before_call returned an awaitable in synchronous execution; use `acall`.",
                tool_name=current_ctx.tool_name,
                policy=policy,
            )
        current_ctx, changed = _apply_decision(policy, current_ctx, decision)
        modified = modified or changed
    return current_ctx, modified


async def aevaluate_before_call(policies: Sequence[Any], ctx: ToolCallContext) -> tuple[ToolCallContext, bool]:
    current_ctx = ctx
    modified = False
    for policy in policies:
        abefore = getattr(policy, "abefore_call", None)
        before = getattr(policy, "before_call", None)
        if callable(abefore):
            decision = abefore(current_ctx)
            if inspect.isawaitable(decision):
                decision = await decision
        elif callable(before):
            decision = before(current_ctx)
            if inspect.isawaitable(decision):
                decision = await decision
        else:
            continue
        current_ctx, changed = _apply_decision(policy, current_ctx, decision)
        modified = modified or changed
    return current_ctx, modified


def evaluate_after_call(policies: Sequence[Any], ctx: ToolCallContext, result: Any) -> Any:
    current = result
    for policy in policies:
        after = getattr(policy, "after_call", None)
        if after is None:
            if callable(getattr(policy, "aafter_call", None)):
                raise ToolPolicyError(
                    f"Async-only policy {type(policy).__name__} cannot run `aafter_call` synchronously; use `acall`.",
                    tool_name=ctx.tool_name,
                    policy=policy,
                )
            continue
        current = after(ctx, current)
        if inspect.isawaitable(current):
            if inspect.iscoroutine(current):
                current.close()
            raise ToolPolicyError(
                f"Policy {type(policy).__name__}.after_call returned an awaitable in synchronous execution; use `acall`.",
                tool_name=ctx.tool_name,
                policy=policy,
            )
    return current


async def aevaluate_after_call(policies: Sequence[Any], ctx: ToolCallContext, result: Any) -> Any:
    current = result
    for policy in policies:
        aafter = getattr(policy, "aafter_call", None)
        after = getattr(policy, "after_call", None)
        if callable(aafter):
            current = aafter(ctx, current)
            if inspect.isawaitable(current):
                current = await current
        elif callable(after):
            current = after(ctx, current)
            if inspect.isawaitable(current):
                current = await current
    return current


def call_with_tool_policy(
    tool_name: str,
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    tool: Tool | None = None,
    annotations: ToolAnnotations | None = None,
    caller: str | None = None,
    invoke: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    """Invoke a plain sync callable mediated by active ``tool_policy``."""
    policies = get_active_tool_policies()
    call_kwargs = dict(kwargs)
    ctx: ToolCallContext | None = None
    if policies:
        ctx = ToolCallContext(
            tool_name=tool_name,
            args=dict(call_kwargs),
            tool=tool,
            annotations=annotations,
            caller=caller if caller is not None else get_tool_caller(),
        )
        ctx, modified = evaluate_before_call(policies, ctx)
        if modified:
            call_kwargs = dict(ctx.args)
    result = invoke(call_kwargs) if invoke is not None else func(**call_kwargs)
    if policies and ctx is not None:
        result = evaluate_after_call(policies, ctx, result)
    return result


async def acall_with_tool_policy(
    tool_name: str,
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    tool: Tool | None = None,
    annotations: ToolAnnotations | None = None,
    caller: str | None = None,
    invoke: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    """Invoke a plain sync or async callable mediated by active ``tool_policy``."""
    policies = get_active_tool_policies()
    call_kwargs = dict(kwargs)
    ctx: ToolCallContext | None = None
    if policies:
        ctx = ToolCallContext(
            tool_name=tool_name,
            args=dict(call_kwargs),
            tool=tool,
            annotations=annotations,
            caller=caller if caller is not None else get_tool_caller(),
        )
        ctx, modified = await aevaluate_before_call(policies, ctx)
        if modified:
            call_kwargs = dict(ctx.args)
    result = invoke(call_kwargs) if invoke is not None else func(**call_kwargs)
    if inspect.isawaitable(result):
        result = await result
    if policies and ctx is not None:
        result = await aevaluate_after_call(policies, ctx, result)
    return result


__all__ = [
    "Allow",
    "Deny",
    "Modify",
    "ReadOnlyToolPolicy",
    "ToolAllowlistPolicy",
    "ToolApprovalPolicy",
    "ToolCallBudgetPolicy",
    "ToolCallContext",
    "ToolPolicy",
    "ToolPolicyDecision",
    "ToolPolicyError",
]
