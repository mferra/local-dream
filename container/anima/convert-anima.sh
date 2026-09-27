#!/usr/bin/env bash
# Converts an Anima checkpoint into a Local Dream QNN package (the zip the app
# imports as a custom NPU model), entirely inside a container.
#
#   container/anima/convert-anima.sh DIT.safetensors TE.safetensors NAME
#
#   DIT  the checkpoint's diffusion model (bf16 release preferred; ComfyUI
#        int8 "convrot" files also load, with their rounding baked in)
#   TE   its Qwen3-0.6B text encoder (Civitai "_txt" file, or the stock
#        qwen_3_06b_base.safetensors when the checkpoint has none)
#   NAME output name -> container/work/NAME/NAME_qnn2.28_8gen3.zip
#
# Output graphs are FP16 for the Hexagon V75 tier ("8gen3": 8 Gen 3, 8 Elite,
# 8 Elite Gen 5), like the published anima-qnn packages. The VAE and the two
# tokenizers are the stock Anima ones and are taken from a published package
# (downloaded once into container/downloads).
#
# Set CHECK=1 to first compare the export model against ComfyUI's Anima
# implementation (fp32 and simulated fp16; slow, ~25 GB RAM).
#
# Status: partly validated. The split DiT matches ComfyUI in fp32 (relative
# error ~3e-7, checked on CyberRealistic Anima v7.0). Not yet verified: the
# text encoder (clip.bin) export, FP16 accuracy, the QNN compile and running
# the package on a device. Test a package on the phone before relying on it.
set -euo pipefail

if [ $# -ne 3 ]; then
    sed -n '2,24p' "$0"
    exit 1
fi
DIT=$(realpath "$1")
TE=$(realpath "$2")
NAME=$3
RESIDUAL_SCALE=${RESIDUAL_SCALE:-32}

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DOWNLOADS=$REPO/container/downloads
WORK=$REPO/container/work
OUT=$WORK/$NAME
RUNTIME=${CONTAINER_RUNTIME:-$(command -v podman || command -v docker)}
IMAGE=local-dream-anima

QNN_ZIP=qnn-2.28.0.241029.zip
REF_ZIP=cyberrealistic_v3_turbo_qnn2.28_8gen3.zip
COMFYUI_COMMIT=4ef23c34d950eecc37040a21ee1741a49d2e44b1

fetch() {
    [ -f "$DOWNLOADS/$1" ] && return
    echo ">> Downloading $1"
    curl -fL --retry 3 -C - -o "$DOWNLOADS/$1.part" "$2"
    mv "$DOWNLOADS/$1.part" "$DOWNLOADS/$1"
}
mkdir -p "$DOWNLOADS" "$WORK"
fetch "$QNN_ZIP" "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/qualcomm_neural_processing_sdk/v2.28.0.241029.zip"
fetch "$REF_ZIP" "https://huggingface.co/xororz/anima-qnn/resolve/main/$REF_ZIP"

echo ">> Building image $IMAGE"
"$RUNTIME" build -q -v "$DOWNLOADS:/cache:ro" -t "$IMAGE" "$HERE" > /dev/null

run() {
    "$RUNTIME" run --rm \
        -v "$HERE:/work:ro" \
        -v "$WORK:/out" \
        -v "$DIT:/in/dit.safetensors:ro" \
        -v "$TE:/in/te.safetensors:ro" \
        -w /work "$IMAGE" "$@"
}

if [ "${CHECK:-0}" = 1 ]; then
    if [ ! -d "$WORK/comfyui/.git" ]; then
        git clone -q https://github.com/comfyanonymous/ComfyUI "$WORK/comfyui"
    fi
    git -C "$WORK/comfyui" -c advice.detachedHead=false checkout -q "$COMFYUI_COMMIT"
    echo ">> Checking against ComfyUI"
    run /opt/venv-ref/bin/python check_reference.py --comfyui /out/comfyui \
        --dit /in/dit.safetensors --te /in/te.safetensors --fp16 \
        --residual_scale "$RESIDUAL_SCALE"
fi

echo ">> Exporting ONNX"
run /opt/venv-ref/bin/python export_onnx.py --dit /in/dit.safetensors \
    --te /in/te.safetensors --out "/out/$NAME/onnx" --residual_scale "$RESIDUAL_SCALE"

PKG=qnn_models_anima_8gen3
for graph in clip unet_part1 unet_part2; do
    echo ">> Converting $graph (FP16, HTP V75)"
    run bash convert_qnn.sh "/out/$NAME/onnx/$graph" "/out/$NAME/output/$PKG" "$graph"
done

echo ">> Packaging"
dest=$OUT/output/$PKG
cp "$OUT/onnx/token_emb.bin" "$dest/"
for f in tokenizer.json tokenizer_t5.json vae_encoder.bin vae_decoder.bin config.json; do
    unzip -q -o -j "$DOWNLOADS/$REF_ZIP" "output/$PKG/$f" -d "$dest"
done
touch "$dest/ANIMA"
rm -f "$OUT/${NAME}_qnn2.28_8gen3.zip"
(cd "$OUT" && zip -q -r "${NAME}_qnn2.28_8gen3.zip" "output/$PKG")
ls -la "$dest"
echo ">> Package: $OUT/${NAME}_qnn2.28_8gen3.zip"
