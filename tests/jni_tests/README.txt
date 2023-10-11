## Build and install tests for arm64 guest

./build/soong/soong_ui.bash --make-mode TARGET_BUILD_VARIANT=userdebug TARGET_PRODUCT=aosp_arm64 berberis_jni_tests

adb install out/target/product/generic_arm64/testcases/berberis_jni_tests/arm64/berberis_jni_tests.apk


## Run tests

adb shell am instrument -w com.berberis.jnitests/androidx.test.runner.AndroidJUnitRunner


## Uninstall tests apk

adb uninstall com.berberis.jnitests
