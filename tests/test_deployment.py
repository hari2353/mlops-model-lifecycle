from mlops_lifecycle.deployment import Tensor, coral_readiness, plan_tensor_arena


def test_non_overlapping_tensors_share_arena_space():
    tensors = [Tensor("a", 1000, 0, 1), Tensor("b", 1000, 2, 3)]
    out = plan_tensor_arena(tensors)
    assert out["peak_bytes"] == 1008  # 1000 rounded up to 16-byte alignment
    assert out["reused_bytes"] == 1008
    assert out["offsets"]["a"] == out["offsets"]["b"] == 0


def test_overlapping_lifetimes_do_not_share_space():
    tensors = [Tensor("a", 100, 0, 5), Tensor("b", 100, 2, 7)]
    out = plan_tensor_arena(tensors)
    assert out["peak_bytes"] == 224
    assert out["offsets"]["a"] != out["offsets"]["b"]


def test_arena_alignment_and_zero_size():
    out = plan_tensor_arena([Tensor("tiny", 1, 0, 0), Tensor("empty", 0, 0, 0)])
    assert out["peak_bytes"] == 16
    out2 = plan_tensor_arena([Tensor("empty", 0, 0, 0)])
    assert out2["peak_bytes"] == 0


def test_arena_is_deterministic_by_name():
    tensors = [Tensor("b", 64, 0, 1), Tensor("a", 64, 0, 1)]
    assert (
        plan_tensor_arena(tensors)["offsets"]
        == plan_tensor_arena(list(reversed(tensors)))["offsets"]
    )


def test_arena_rejects_invalid_tensors():
    for tensors in (
        [Tensor("a", -1, 0, 1)],
        [Tensor("a", 1, -1, 1)],
        [Tensor("a", 1, 2, 1)],
        [Tensor("a", 1, 0, 1), Tensor("a", 1, 0, 1)],
    ):
        try:
            plan_tensor_arena(tensors, alignment=16)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid tensors: {tensors}")
    try:
        plan_tensor_arena([], alignment=0)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted zero alignment")


def test_coral_ready_for_integer_model():
    out = coral_readiness("int8", "int8", ["conv2d", "fully_connected"])
    assert out["ready"] is True
    assert out["blocking"] == []


def test_coral_blocks_float_io():
    out = coral_readiness("float32", "int8", ["conv2d"])
    assert out["ready"] is False
    assert "input dtype" in out["blocking"][0]


def test_coral_blocks_unsupported_ops():
    out = coral_readiness(
        "int8", "uint8", ["conv2d", "custom_op"], unsupported_ops={"custom_op"}
    )
    assert out["ready"] is False
    assert out["blocking"] == ["unsupported ops: custom_op"]


def test_coral_rejects_empty_op_list():
    try:
        coral_readiness("int8", "int8", [])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an empty op list")
