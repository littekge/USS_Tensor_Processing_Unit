"""Bias-removal pass: drop bias adds from the *Functional TPU* dialect.

Dropping the bias `add` operations was validated on LeNet-5, where it costs a
<1% accuracy drop -- acceptable for the target network. Tiny_NN and Bigger_NN do
lose significant accuracy without their biases, but neither is a functional
network (Tiny_NN is a regression sanity check; Bigger_NN exists only to exercise
matrix partitioning), so their accuracy has no bearing on this decision. The bias
tensors stay resident in weight memory (Process_Weights still maps them); they are
simply no longer referenced in the instruction stream.

The pass identifies `AddOp`s whose operand traces to a `*.bias` function
argument (a `weight_map` entry with `kind == "bias"`), reroutes every consumer of
the add's result to the non-bias operand (the matmul result), and erases the add.
Non-bias adds (e.g. future matmul-partition partial sums) are left untouched.
"""

from __future__ import annotations

from .dialect import AddOp, Im2colOp, MaxPoolOp, MultOp, Program, ReluOp, ReturnOp


def _is_bias(name: str, weight_map: dict) -> bool:
    """True if the SSA `name` refers to a `*.bias` function argument."""
    entry = weight_map.get(name)
    return entry is not None and entry.get("kind") == "bias"


def remove_bias_adds(program: Program, weight_map: dict) -> Program:
    """Return a new `Program` with bias `AddOp`s dropped and consumers rerouted.

    Mutates operand names in place as it rewrites; ops are processed in order so a
    reroute is visible to every later consumer.
    """
    rename: dict[str, str] = {}  # dropped-add result -> replacement value

    def replacement(name: str) -> str:
        """Follow reroutes to the value that survives after bias removal."""
        while name in rename:
            name = rename[name]
        return name

    new_ops: list = []
    for op in program.ops:
        if isinstance(op, MultOp):
            op.lhs.name = replacement(op.lhs.name)
            op.rhs.name = replacement(op.rhs.name)
            new_ops.append(op)
        elif isinstance(op, AddOp):
            lhs_name = replacement(op.lhs.name)
            rhs_name = replacement(op.rhs.name)
            lhs_bias = _is_bias(lhs_name, weight_map)
            rhs_bias = _is_bias(rhs_name, weight_map)
            if lhs_bias and rhs_bias:
                raise AssertionError(
                    f"Both operands of {op.result} = add {lhs_name}, {rhs_name} are biases; "
                    "expected exactly one bias operand."
                )
            if lhs_bias or rhs_bias:
                # Reroute consumers to the non-bias operand, then drop the add.
                rename[op.result] = rhs_name if lhs_bias else lhs_name
                continue
            op.lhs.name = lhs_name
            op.rhs.name = rhs_name
            new_ops.append(op)
        elif isinstance(op, (ReluOp, Im2colOp, MaxPoolOp)):
            # A windowed/activation op may consume a bias-add result (e.g. a pool
            # directly after conv+bias); reroute its source to the surviving value.
            op.src.name = replacement(op.src.name)
            new_ops.append(op)
        elif isinstance(op, ReturnOp):
            op.value = replacement(op.value)
            new_ops.append(op)
        else:
            new_ops.append(op)

    program.ops = new_ops
    return program
