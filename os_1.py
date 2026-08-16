import torch
import numpy as np

from src.chronos.chronos2.pipeline import Chronos2Pipeline


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("Environment")
print("=" * 70)
print("PyTorch :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Test device:", DEVICE)

if not torch.cuda.is_available():
    print("\nWARNING: CUDA is not available.")
    print("CUDA-specific tests will be skipped.")


# ============================================================
# Helper
# ============================================================

def run_test(name, fn):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    try:
        result = fn()
        print("PASS")
        return result

    except Exception as e:
        print("FAIL")
        print("Error:", type(e).__name__)
        print("Message:", e)
        return None


def print_embeddings(result):
    """
    Print the structure returned by pipeline.embed().
    """
    embeds, loc_scales = result

    print("embeds type:", type(embeds))
    print("number of embeddings:", len(embeds))

    for i, e in enumerate(embeds):
        print(
            f"  embedding[{i}]: "
            f"type={type(e)}, "
            f"shape={e.shape}, "
            f"dtype={e.dtype}, "
            f"device={e.device}"
        )

    print("loc_scales type:", type(loc_scales))
    print("number of loc_scales:", len(loc_scales))

    for i, (loc, scale) in enumerate(loc_scales):
        print(
            f"  loc_scales[{i}]: "
            f"loc shape={loc.shape}, "
            f"scale shape={scale.shape}"
        )


# ============================================================
# 1. CPU model + CPU Tensor
# ============================================================

cpu_pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)

print("\nCPU model device:", cpu_pipeline.model.device)


def test_cpu_model_cpu_tensor():
    x = torch.rand(2, 6, 60)

    print("Input:", type(x), x.shape, x.device)

    result = cpu_pipeline.embed(x)

    print_embeddings(result)

    assert len(result[0]) == 2

    return result


run_test(
    "TEST 1: CPU model + CPU Tensor",
    test_cpu_model_cpu_tensor,
)


# ============================================================
# CUDA tests
# ============================================================

if torch.cuda.is_available():

    cuda_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda",
    )

    print("\nCUDA model device:", cuda_pipeline.model.device)

    # --------------------------------------------------------
    # 2. CUDA model + CPU Tensor
    # --------------------------------------------------------

    def test_cuda_model_cpu_tensor():

        x = torch.rand(2, 6, 60)

        print("Input device:", x.device)
        print("Model device:", cuda_pipeline.model.device)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 2: CUDA model + CPU Tensor",
        test_cuda_model_cpu_tensor,
    )


    # --------------------------------------------------------
    # 3. CUDA model + CUDA Tensor
    # ORIGINAL BUG
    # --------------------------------------------------------

    def test_cuda_model_cuda_tensor():

        x = torch.rand(
            2,
            6,
            60,
            device="cuda",
        )

        print("Input device:", x.device)
        print("Model device:", cuda_pipeline.model.device)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    cuda_tensor_result = run_test(
        "TEST 3: CUDA model + CUDA Tensor ⭐ ORIGINAL BUG",
        test_cuda_model_cuda_tensor,
    )


    # --------------------------------------------------------
    # 4. List of CPU tensors
    # --------------------------------------------------------

    def test_list_cpu_tensors():

        x = [
            torch.rand(6, 60),
            torch.rand(6, 60),
        ]

        print("Input type:", type(x))
        print("Element 0:", type(x[0]), x[0].device)
        print("Element 1:", type(x[1]), x[1].device)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 4: CUDA model + List[CPU Tensor]",
        test_list_cpu_tensors,
    )


    # --------------------------------------------------------
    # 5. List of CUDA tensors
    # --------------------------------------------------------

    def test_list_cuda_tensors():

        x = [
            torch.rand(6, 60, device="cuda"),
            torch.rand(6, 60, device="cuda"),
        ]

        print("Input type:", type(x))
        print("Element 0:", type(x[0]), x[0].device)
        print("Element 1:", type(x[1]), x[1].device)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 5: CUDA model + List[CUDA Tensor] ⭐",
        test_list_cuda_tensors,
    )


    # --------------------------------------------------------
    # 6. NumPy array
    # --------------------------------------------------------

    def test_numpy_array():

        x = np.random.rand(2, 6, 60)

        print("Input type:", type(x))
        print("Input shape:", x.shape)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 6: CUDA model + NumPy array",
        test_numpy_array,
    )


    # --------------------------------------------------------
    # 7. List of NumPy arrays
    # --------------------------------------------------------

    def test_list_numpy():

        x = [
            np.random.rand(6, 60),
            np.random.rand(6, 60),
        ]

        print("Input type:", type(x))
        print("Element 0:", type(x[0]))
        print("Element 1:", type(x[1]))

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 7: CUDA model + List[NumPy arrays]",
        test_list_numpy,
    )


    # --------------------------------------------------------
    # 8. Univariate input
    # --------------------------------------------------------

    def test_univariate():

        x = torch.rand(
            2,
            1,
            60,
            device="cuda",
        )

        print("Input shape:", x.shape)
        print("Input device:", x.device)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 8: CUDA model + CUDA Tensor, n_variates=1",
        test_univariate,
    )


    # --------------------------------------------------------
    # 9. More variates
    # --------------------------------------------------------

    def test_many_variates():

        x = torch.rand(
            2,
            10,
            60,
            device="cuda",
        )

        print("Input shape:", x.shape)
        print("Input device:", x.device)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 9: CUDA model + CUDA Tensor, n_variates=10",
        test_many_variates,
    )


    # --------------------------------------------------------
    # 10. Different history lengths
    # --------------------------------------------------------

    def test_different_history_lengths():

        x = [
            torch.rand(6, 40, device="cuda"),
            torch.rand(6, 60, device="cuda"),
            torch.rand(6, 80, device="cuda"),
        ]

        print("Input shapes:")
        for i, item in enumerate(x):
            print(i, item.shape, item.device)

        result = cuda_pipeline.embed(x)

        print_embeddings(result)

        assert len(result[0]) == 3

        return result


    run_test(
        "TEST 10: CUDA model + CUDA tensors with different history lengths",
        test_different_history_lengths,
    )


    # --------------------------------------------------------
    # 11. Explicit small batch size
    # --------------------------------------------------------

    def test_small_batch_size():

        x = torch.rand(
            20,
            6,
            60,
            device="cuda",
        )

        print("Input shape:", x.shape)

        result = cuda_pipeline.embed(
            x,
            batch_size=4,
        )

        print_embeddings(result)

        assert len(result[0]) == 20

        return result


    run_test(
        "TEST 11: CUDA input + batch_size=4",
        test_small_batch_size,
    )


    # --------------------------------------------------------
    # 12. More samples than default batch size
    # --------------------------------------------------------

    def test_multiple_batches():

        x = torch.rand(
            300,
            6,
            60,
            device="cuda",
        )

        print("Input shape:", x.shape)

        result = cuda_pipeline.embed(
            x,
            batch_size=64,
        )

        print("Number of returned embeddings:", len(result[0]))

        assert len(result[0]) == 300

        return result


    run_test(
        "TEST 12: CUDA input + multiple DataLoader batches",
        test_multiple_batches,
    )


    # --------------------------------------------------------
    # 13. Context length
    # --------------------------------------------------------

    def test_context_length():

        x = torch.rand(
            2,
            6,
            60,
            device="cuda",
        )

        print("Input shape:", x.shape)

        result = cuda_pipeline.embed(
            x,
            context_length=40,
        )

        print_embeddings(result)

        assert len(result[0]) == 2

        return result


    run_test(
        "TEST 13: CUDA input + explicit context_length",
        test_context_length,
    )


    # --------------------------------------------------------
    # 14. CPU vs CUDA output comparison
    # --------------------------------------------------------

    def test_cpu_cuda_equivalence():

        # Create data ONCE on CPU
        x_cpu = torch.rand(
            2,
            6,
            60,
        )

        # Same exact values on CUDA
        x_cuda = x_cpu.cuda()

        cpu_result = cuda_pipeline.embed(x_cpu)[0]
        cuda_result = cuda_pipeline.embed(x_cuda)[0]

        print("Comparing CPU-input and CUDA-input embeddings")

        for i, (a, b) in enumerate(
            zip(cpu_result, cuda_result)
        ):
            same = torch.allclose(
                a,
                b,
                atol=1e-5,
                rtol=1e-5,
            )

            max_difference = torch.max(
                torch.abs(a - b)
            ).item()

            print(
                f"Sample {i}: "
                f"same={same}, "
                f"max_difference={max_difference}"
            )

            assert same

        return True


    run_test(
        "TEST 14: CPU input vs CUDA input output equivalence ⭐",
        test_cpu_cuda_equivalence,
    )


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("TESTING COMPLETE")
print("=" * 70)