"""Small self-contained ONNX graphs and integrity-preserving hostile bundles."""

import onnx
from onnx import TensorProto as T, helper as h

from mlbundle import MemberName
from mlbundle.container import read_container
from mlbundle.writing import write_container

ROWS = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]


def model(nodes, inputs, outputs, *, initializers=(), value_info=()):
    graph = h.make_graph(
        nodes,
        "tabular",
        inputs,
        outputs,
        list(initializers),
        value_info=list(value_info),
    )
    return h.make_model(
        graph,
        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("ai.onnx.ml", 3)],
        ir_version=10,
        producer_name="fixture",
        producer_version="1",
    )


def classifier(labels=(10, 20), *, zipmap=False, map_only=False):
    strings = isinstance(labels[0], str)
    label_attr = {
        "classlabels_strings" if strings else "classlabels_ints": list(labels)
    }
    nodes = [
        h.make_node(
            "LinearClassifier",
            ["X"],
            ["label", "P"],
            domain="ai.onnx.ml",
            coefficients=[-1.0, 1.0, 1.0, -1.0],
            intercepts=[0.0, 0.0],
            post_transform="SOFTMAX",
            **label_attr,
        )
    ]
    outputs = [
        h.make_tensor_value_info("label", T.STRING if strings else T.INT64, [None])
    ]
    if zipmap:
        attrs = {
            "classlabels_strings" if strings else "classlabels_int64s": list(labels)
        }
        nodes.append(
            h.make_node(
                "ZipMap", ["P"], ["probabilities"], domain="ai.onnx.ml", **attrs
            )
        )
        typ = h.make_sequence_type_proto(
            h.make_map_type_proto(
                T.STRING if strings else T.INT64, h.make_tensor_type_proto(T.FLOAT, [])
            )
        )
        outputs.append(h.make_value_info("probabilities", typ))
    else:
        outputs.append(h.make_tensor_value_info("P", T.FLOAT, [None, 2]))
    return model(
        nodes,
        [h.make_tensor_value_info("X", T.FLOAT, [None, 2])],
        outputs[1:] if map_only else outputs,
    )


def softmax(*, two_outputs=False):
    nodes = [
        h.make_node("MatMul", ["X", "W"], ["scores"]),
        h.make_node("Add", ["scores", "B"], ["logits"]),
        h.make_node("Softmax", ["logits"], ["P"], axis=1),
    ]
    outputs = [h.make_tensor_value_info("P", T.FLOAT, [None, 2])]
    if two_outputs:
        nodes.append(h.make_node("ArgMax", ["P"], ["label"], axis=1, keepdims=0))
        outputs.insert(0, h.make_tensor_value_info("label", T.INT64, [None]))
    return model(
        nodes,
        [h.make_tensor_value_info("X", T.FLOAT, [None, 2])],
        outputs,
        initializers=[
            h.make_tensor("W", T.FLOAT, [2, 2], [-1, 1, 1, -1]),
            h.make_tensor("B", T.FLOAT, [2], [0, 0]),
        ],
    )


def regressor(*, columns=False, batch=None):
    if columns:
        inputs = [
            h.make_tensor_value_info("a", T.INT64, [batch, 1]),
            h.make_tensor_value_info("b", T.DOUBLE, [batch, 1]),
        ]
        nodes = [
            h.make_node("Cast", ["a"], ["a_float"], to=T.FLOAT),
            h.make_node("Cast", ["b"], ["b_float"], to=T.FLOAT),
            h.make_node("Concat", ["a_float", "b_float"], ["X"], axis=1),
        ]
    else:
        inputs = [h.make_tensor_value_info("X", T.FLOAT, [batch, 2])]
        nodes = []
    nodes.append(
        h.make_node(
            "LinearRegressor",
            ["X"],
            ["Y"],
            domain="ai.onnx.ml",
            coefficients=[2.0, -1.0],
            intercepts=[0.5],
            targets=1,
        )
    )
    return model(nodes, inputs, [h.make_tensor_value_info("Y", T.FLOAT, [batch, 1])])


def rewrite(path, change, *, schema=False):
    manifest, members = read_container(path)
    parts = manifest.model_dump(mode="json", by_alias=True)
    del members[MemberName.MANIFEST]
    change(parts, members)
    if schema:
        from conftest import align_schema

        align_schema(parts, members)
    return write_container(parts, members)


def external_tensor(location="outside.bin", *, entries=True):
    tensor = onnx.TensorProto(
        name="external", data_type=T.FLOAT, dims=[1], data_location=T.EXTERNAL
    )
    if entries:
        tensor.external_data.add(key="location", value=str(location))
    return tensor


def external_model(where, location="outside.bin"):
    native = regressor()
    tensor = external_tensor(location)
    if where == "initializer":
        native.graph.initializer.append(tensor)
    elif where == "attribute":
        native.graph.node.insert(
            0, h.make_node("Constant", [], ["unused"], value=tensor)
        )
    elif where == "subgraph":
        nested = h.make_graph(
            [h.make_node("Constant", [], ["external"], value=tensor)],
            "nested",
            [],
            [h.make_tensor_value_info("external", T.FLOAT, [1])],
        )
        native.graph.node.insert(
            0,
            h.make_node(
                "If", ["condition"], ["unused"], then_branch=nested, else_branch=nested
            ),
        )
    elif where == "sparse":
        sparse = onnx.SparseTensorProto(
            values=tensor, indices=h.make_tensor("indices", T.INT64, [1], [0]), dims=[1]
        )
        native.graph.sparse_initializer.append(sparse)
    elif where == "function":
        function = h.make_function(
            "ai.onnx.ml",
            "Hidden",
            [],
            ["external"],
            [h.make_node("Constant", [], ["external"], value=tensor)],
            [h.make_opsetid("", 21)],
        )
        native.functions.append(function)
    return native
