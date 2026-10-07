#!/usr/bin/env python3
"""
Fix ONNX model type issues, particularly for Gather operations.

This script fixes dtype mismatches where float tensors are used as indices
for Gather operations, which require int64 indices.
"""

import onnx
from onnx import helper, TensorProto
import sys


def make_prediction_length_dynamic(model: onnx.ModelProto, dim_name: str = "prediction_length"):
    """
    Make the prediction_length dimension (dim 2) of the output dynamic.

    Changes output shape from [batch_size, num_quantiles, 64] to [batch_size, num_quantiles, prediction_length]
    where prediction_length is a symbolic dimension.
    """
    print("\nMaking prediction_length dimension dynamic...")

    # Update output tensor shapes
    for output in model.graph.output:
        if output.type.tensor_type.HasField("shape"):
            shape = output.type.tensor_type.shape
            # Check if this is the quantile_preds output (3D tensor: [batch, quantiles, pred_len])
            if len(shape.dim) == 3:
                print(f"  Output '{output.name}' shape before:")
                for i, dim in enumerate(shape.dim):
                    if dim.HasField("dim_value"):
                        print(f"    Dim {i}: {dim.dim_value}")
                    elif dim.HasField("dim_param"):
                        print(f"    Dim {i}: {dim.dim_param} (symbolic)")

                # Make dimension 2 (prediction_length) dynamic
                if shape.dim[2].HasField("dim_value"):
                    original_value = shape.dim[2].dim_value
                    shape.dim[2].Clear()
                    shape.dim[2].dim_param = dim_name
                    print(f"  Changed dim 2 from {original_value} to '{dim_name}' (dynamic)")

    return model


def fix_gather_indices(model_path: str, output_path: str, make_dynamic: bool = True):
    """
    Fix Gather operation index type issues in ONNX model and optionally make prediction_length dynamic.

    The indices may be represented as float tensors in the graph but Gather
    requires int64. This function inserts Cast operations to convert float
    indices to int64 before Gather operations.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save fixed ONNX model
        make_dynamic: If True, also make the prediction_length dimension dynamic
    """
    print(f"Loading ONNX model from {model_path}")
    model = onnx.load(model_path)

    # Find all Gather nodes and check their index inputs
    gather_nodes = []

    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Gather":
            gather_nodes.append((idx, node))
            if len(node.input) >= 2:
                index_input = node.input[1]
                print(f"Gather node {node.name or 'unnamed'} uses indices: {index_input}")

    print(f"\nFound {len(gather_nodes)} Gather operations")

    # Insert Cast nodes before Gather operations to convert float indices to int64
    print("\nInserting Cast operations for float->int64 conversion...")
    cast_count = 0

    for idx, gather_node in gather_nodes:
        if len(gather_node.input) < 2:
            continue

        index_input = gather_node.input[1]

        # Create a unique name for the cast output
        cast_output_name = f"{index_input}_int64_cast"

        # Create Cast node: float -> int64
        cast_node = helper.make_node(
            "Cast",
            inputs=[index_input],
            outputs=[cast_output_name],
            to=TensorProto.INT64,
            name=f"cast_{index_input}_to_int64",
        )

        # Modify the Gather node to use the cast output
        new_gather_input = [gather_node.input[0], cast_output_name]
        if len(gather_node.input) > 2:
            new_gather_input.extend(gather_node.input[2:])

        # Update the gather node's inputs
        del gather_node.input[:]
        gather_node.input.extend(new_gather_input)

        # Add the cast node before this gather node
        model.graph.node.insert(idx + cast_count, cast_node)
        cast_count += 1

        print(f"  Added Cast node before {gather_node.name or 'unnamed'}")

    print(f"Added {cast_count} Cast operations before Gather nodes")

    # Fix Concat operations that might have dtype mismatches
    # Cast all int64 inputs back to float32 before Concat
    print("\nFixing Concat operations with dtype mismatches...")
    concat_cast_count = 0

    concat_nodes = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Concat":
            concat_nodes.append((idx, node))

    print(f"Found {len(concat_nodes)} Concat operations")

    for idx, concat_node in concat_nodes:
        # For each Concat input that might be int64, cast it back to float32
        new_inputs = []
        for i, input_name in enumerate(concat_node.input):
            # Check if this input came from a Cast operation (has "_int64_cast" in name)
            if "_int64_cast" in input_name:
                # This was cast to int64 for Gather, need to cast back to float for Concat
                cast_output_name = f"{input_name}_back_to_float32"

                cast_node = helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_output_name],
                    to=TensorProto.FLOAT,
                    name=f"cast_{input_name}_back_to_float",
                )

                # Insert cast node before concat
                model.graph.node.insert(idx + concat_cast_count, cast_node)
                concat_cast_count += 1

                new_inputs.append(cast_output_name)
                print(f"  Adding Cast int64→float32 before Concat {concat_node.name or 'unnamed'} input {i}")
            else:
                new_inputs.append(input_name)

        # Update concat inputs
        if new_inputs != list(concat_node.input):
            del concat_node.input[:]
            concat_node.input.extend(new_inputs)

    print(f"Added {concat_cast_count} Cast operations before Concat nodes")

    # Make prediction_length dimension dynamic
    if make_dynamic:
        model = make_prediction_length_dynamic(model)

    # Validate and save
    print("\nValidating fixed model...")
    try:
        onnx.checker.check_model(model)
        print("✓ Model validation passed!")
    except Exception as e:
        print(f"⚠ Validation warnings: {e}")
        print("  Attempting to save anyway...")

    print(f"\nSaving fixed model to {output_path}")
    onnx.save(model, output_path)
    print("✓ Model saved successfully!")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix ONNX model type issues")
    parser.add_argument("input", help="Input ONNX model path")
    parser.add_argument("output", help="Output ONNX model path")

    args = parser.parse_args()

    try:
        fix_gather_indices(args.input, args.output)
        print("\n✓ Model fixed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
