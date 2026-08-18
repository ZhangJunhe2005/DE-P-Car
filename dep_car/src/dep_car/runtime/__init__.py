"""Runtime-only P6 adapters around the immutable P4/P5 policy contract.

Keep package import lightweight: the system-Python safety node has no PyTorch,
while only the separate yopo policy process imports :mod:`p6_policy`.
"""
