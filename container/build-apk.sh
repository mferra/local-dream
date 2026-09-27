#!/usr/bin/env bash
# Builds the Local Dream APK entirely inside a container (podman or docker):
# the native core (libstable_diffusion_core.so + QNN libraries) and then the
# APK. Nothing but the container runtime is needed on the host.
#
#   container/build-apk.sh              # basic (no NSFW filter), release
#   FLAVOR=filter container/build-apk.sh
#
# Host-side state:
#   container/downloads/              downloaded SDK/NDK archives, git-ignored:
#                                     their licenses forbid redistributing them,
#                                     so only their checksums are versioned
#                                     (container/downloads.sha256)
#   ~/.config/local-dream-signing     signing keystore + its passwords; keep a
#                                     backup: updates to the installed app must
#                                     be signed with the same key
#   volumes local-dream-gradle/-cargo  Gradle and cargo download caches
#
# The DiT engine (Z-Image / FLUX.2 / Qwen, SM8750+ only) is not built here.
set -euo pipefail

FLAVOR=${FLAVOR:-basic}
case "$FLAVOR" in
    basic | filter) ;;
    *) echo "FLAVOR must be basic or filter" >&2; exit 1 ;;
esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
CACHE=${LOCAL_DREAM_CACHE:-$REPO/container/downloads}
SIGNING=${LOCAL_DREAM_SIGNING_DIR:-$HOME/.config/local-dream-signing}
IMAGE=local-dream-builder
RUNTIME=${CONTAINER_RUNTIME:-$(command -v podman || command -v docker)}

NDK_ZIP=android-ndk-r28c-linux.zip
CMDLINE_TOOLS_ZIP=commandlinetools-linux-16111833_latest.zip
QNN_VERSION=2.50.0.260828
QNN_ZIP=qairt-$QNN_VERSION.zip

fetch() {
    local file=$1 url=$2
    [ -f "$CACHE/$file" ] && return
    echo ">> Downloading $file"
    curl -fL --retry 3 -C - -o "$CACHE/$file.part" "$url"
    mv "$CACHE/$file.part" "$CACHE/$file"
}

mkdir -p "$CACHE"
fetch "$NDK_ZIP" "https://dl.google.com/android/repository/$NDK_ZIP"
fetch "$CMDLINE_TOOLS_ZIP" "https://dl.google.com/android/repository/$CMDLINE_TOOLS_ZIP"
fetch "$QNN_ZIP" "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$QNN_VERSION/v$QNN_VERSION.zip"

echo ">> Verifying archives"
(cd "$CACHE" && sha256sum --check --strict "$REPO/container/downloads.sha256")

echo ">> Building image $IMAGE"
"$RUNTIME" build \
    -v "$CACHE:/cache:ro" \
    --build-arg NDK_ZIP="$NDK_ZIP" \
    --build-arg CMDLINE_TOOLS_ZIP="$CMDLINE_TOOLS_ZIP" \
    --build-arg QNN_VERSION="$QNN_VERSION" \
    -t "$IMAGE" "$REPO/container"

if [ ! -f "$SIGNING/keystore.jks" ]; then
    echo ">> Creating signing key in $SIGNING"
    mkdir -p "$SIGNING"
    chmod 700 "$SIGNING"
    password=$(head -c 24 /dev/urandom | base64 | tr -d '/+=')
    "$RUNTIME" run --rm -v "$SIGNING:/signing" "$IMAGE" \
        keytool -genkeypair -keystore /signing/keystore.jks -storetype PKCS12 \
        -alias localdream -keyalg RSA -keysize 4096 -validity 10000 \
        -dname "CN=Local Dream (personal build)" \
        -storepass "$password" -keypass "$password"
    # Gradle maps ORG_GRADLE_PROJECT_<name> variables to project properties.
    cat > "$SIGNING/signing.env" <<EOF
ORG_GRADLE_PROJECT_RELEASE_STORE_FILE=/signing/keystore.jks
ORG_GRADLE_PROJECT_RELEASE_STORE_PASSWORD=$password
ORG_GRADLE_PROJECT_RELEASE_KEY_ALIAS=localdream
ORG_GRADLE_PROJECT_RELEASE_KEY_PASSWORD=$password
EOF
    chmod 600 "$SIGNING/keystore.jks" "$SIGNING/signing.env"
fi

variant="$(tr '[:lower:]' '[:upper:]' <<< "${FLAVOR:0:1}")${FLAVOR:1}"
echo ">> Building native core and ${FLAVOR}Release APK"
"$RUNTIME" run --rm \
    -v "$REPO:/src" \
    -v "$SIGNING:/signing:ro" \
    -v local-dream-gradle:/root/.gradle \
    -v local-dream-cargo:/opt/cargo/registry \
    --env-file "$SIGNING/signing.env" \
    "$IMAGE" \
    bash -euo pipefail -c "
        (cd app/src/main/cpp && bash build.sh)
        ./gradlew --no-daemon assemble${variant}Release
    "

echo ">> APK:"
ls -1 "$REPO"/app/build/outputs/apk/"$FLAVOR"/release/*.apk
