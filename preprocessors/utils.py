import onnxscript
import numpy as np


def pad_list(arrays, axis=0):
    lens = np.array([array.shape[axis] for array in arrays])
    max_len = lens.max()

    def pads(array):
        return [(0, max_len - array.shape[axis]) if i == axis else (0, 0) for i in range(array.ndim)]

    return np.stack([np.pad(array, pads(array)) for array in arrays]), lens


def save_model(function: onnxscript.OnnxFunction, filename: str):
    model = function.to_model_proto()
    model = onnxscript.optimizer.optimize(model)
    model = onnxscript.ir.from_proto(model)
    model = onnxscript.optimizer.optimize(model)

    model.producer_name = "OnnxScript"
    model.producer_version = onnxscript.__version__
    model.metadata_props["model_author"] = "Ilya Stupakov"
    model.metadata_props["model_license"] = "MIT License"
    onnxscript.ir.save(model, filename)
