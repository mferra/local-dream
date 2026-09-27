#!/usr/bin/env bash
# ONNX -> FP16 QNN context binary for the Hexagon V75 ("8gen3" tier, which
# also runs on 8 Elite / 8 Elite Gen 5), the way the published Anima
# packages are built: float graph, fp32 I/O kept (the app memcpys fp32 and
# int32 buffers straight into the tensors), no quantization.
#
#   convert_qnn.sh <onnx dir> <output dir> <binary name>
set -euo pipefail
src=$1 out=$2 name=$3

export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang
export PATH=/opt/venv-qnn/bin:$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH

qairt-converter --onnx_no_simplification \
    --input_network "$src/model.onnx" \
    --output_path "$src/model.dlc" \
    --float_bitwidth 16 \
    --preserve_io_datatype

mkdir -p "$out"
qnn-context-binary-generator \
    --dlc_path "$src/model.dlc" \
    --model "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file /work/htp_backend_8gen3.json \
    --output_dir "$out" \
    --binary_file "$name"
