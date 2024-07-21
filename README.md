# Berberis

Dynamic binary translator to run Android apps with riscv64 native code on x86_64 devices or emulators.

Supported extensions include Zb* (bit manipulation) and most of Zv (vector). Some less commonly used vector instructions are not yet implemented, but Android CTS and some Android apps run with the current set of implemented instructions.

## Getting Started

Note: Googlers, read go/berberis and go/berberis-start first.

### Build

From your Android root checkout, run:

```
source build/envsetup.sh
lunch sdk_phone64_x86_64_riscv64-trunk_staging-eng
m berberis_all
```

For development, we recommend building all existing targets before uploading changes, since they are currently not always synchronized with `berberis_all`:

```
mmm frameworks/libs/binary_translation
```

### Run Hello World

```
out/host/linux-x86/bin/berberis_program_runner_riscv64 \
out/target/product/emu64xr/testcases/berberis_hello_world_static.native_bridge/x86_64/berberis_hello_world_static
```

On success `Hello!` will be printed.

### Run unit tests on host

```
m berberis_all berberis_run_host_tests
```

or

```
out/host/linux-x86/nativetest64/berberis_host_tests/berberis_host_tests
```

### Build and run emulator with Berberis support

```
m
emulator -memory 4096 -writable-system -partition-size 65536 -qemu -cpu host &
```

### Run unit tests on device or emulator

Note: Requires a running device or emulator with Berberis support.

1. Sync tests to the device:

```
adb root
adb sync data
```

2. Run guest loader tests:

```
adb shell /data/nativetest64/berberis_guest_loader_riscv64_tests/berberis_guest_loader_riscv64_tests
```

3. Run program tests:

```
adb shell /data/nativetest64/berberis_ndk_program_tests/berberis_ndk_program_tests
```

## Bionic unit tests

Note: Requires a running device or emulator with Berberis support.

1. Build Bionic unit tests:

```
m TARGET_PRODUCT=aosp_riscv64 bionic-unit-tests
```

2. Push tests to emulator or device:

```
adb push out/target/product/generic_riscv64/data/nativetest64/bionic-loader-test-libs /data/local/tmp
adb push out/target/product/generic_riscv64/data/nativetest64/bionic-unit-tests /data/local/tmp
```

3. Run Bionic tests:

```
adb shell /system/bin/berberis_program_runner_riscv64 /data/local/tmp/bionic-unit-tests/bionic-unit-tests --no_isolate
```
