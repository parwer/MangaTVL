import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

MODELS = Path(__file__).resolve().parents[1] / "manga_translator" / "assets" / "models"
SRC = MODELS / "comictextdetector.pt.onnx"
DST = MODELS / "comictextdetector_subpixel.onnx"


def phase_convs(w: np.ndarray, k: int, pad: int):
    """ConvTranspose (stride 2) weights (Cin, Cout, k, k) split into the 4 output phases (a, b): each a
    stride-1 Conv with only its own taps, as (a, b, weight, pads)."""
    if k == 2 and pad == 0:  # y[2i+a] = x[i] w[a]: 1x1 conv per phase
        return [(a, b, np.ascontiguousarray(w[:, :, a:a + 1, b:b + 1].transpose(1, 0, 2, 3)), [0, 0, 0, 0])
                for a in range(2) for b in range(2)]
    assert k == 4 and pad == 1
    # y[2i+a] = x[i-1] w[3] + x[i] w[1] (a=0);  x[i] w[2] + x[i+1] w[0] (a=1)
    taps = {0: [3, 1], 1: [2, 0]}            # kernel taps for input offsets (-1, 0) / (0, +1)
    out = []
    for a in range(2):
        for b in range(2):
            wc = w[:, :, taps[a]][:, :, :, taps[b]].transpose(1, 0, 2, 3)
            pads = [1 - a, 1 - b, a, b]       # top, left, bottom, right
            out.append((a, b, np.ascontiguousarray(wc), pads))
    return out


def convert(src: Path = SRC, dst: Path = DST):
    """Rewrite every stride-2 ConvTranspose as 4 phase Convs + DepthToSpace (same output, same work;
    onnxruntime's CPU ConvTranspose took 18 of the model's 25 s per page)."""
    m = onnx.load(str(src))
    init = {i.name: i for i in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        if n.op_type != "ConvTranspose":
            nodes.append(n)
            continue
        a_ = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        assert a_["strides"] == [2, 2] and a_.get("group", 1) == 1 and not a_.get("output_padding")
        w = numpy_helper.to_array(init[n.input[1]])
        bias = numpy_helper.to_array(init[n.input[2]]) if len(n.input) > 2 and n.input[2] else None
        phases = []
        for a, b, wc, pads in phase_convs(w, a_["kernel_shape"][0], a_["pads"][0]):
            wname, out = f"{n.name}_p{a}{b}_w", f"{n.output[0]}_p{a}{b}"
            m.graph.initializer.append(numpy_helper.from_array(wc, wname))
            inputs = [n.input[0], wname]
            if bias is not None:
                m.graph.initializer.append(numpy_helper.from_array(bias, f"{n.name}_p{a}{b}_b"))
                inputs.append(f"{n.name}_p{a}{b}_b")
            nodes.append(helper.make_node("Conv", inputs, [out], name=f"{n.name}_p{a}{b}",
                                          kernel_shape=list(wc.shape[2:]), pads=pads, strides=[1, 1]))
            phases.append(out)
        # channels phase-major (a, b, c): DepthToSpace DCR puts phase (a, b) at (2i+a, 2j+b)
        nodes.append(helper.make_node("Concat", phases, [n.output[0] + "_phases"], name=n.name + "_cat", axis=1))
        nodes.append(helper.make_node("DepthToSpace", [n.output[0] + "_phases"], [n.output[0]], name=n.name + "_d2s",
                                      blocksize=2, mode="DCR"))
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    used = {i for n in m.graph.node for i in n.input}
    keep = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return dst


if __name__ == "__main__":
    print(convert(*(Path(p) for p in sys.argv[1:3])))
