#
# Copyright (C) 2023 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file defines:
#   BERBERIS_PRODUCT_PACKAGES - list of main product packages
#   BERBERIS_DEV_PRODUCT_PACKAGES - list of development packages
#

include frameworks/libs/binary_translation/riscv64_to_x86_64_config.mk

PRODUCT_PACKAGES += $(BERBERIS_PRODUCT_PACKAGES)

# ATTENTION: we are overriding
# PRODUCT_SYSTEM_PROPERTIES += ro.dalvik.vm.native.bridge?=0
# set by build/make/target/product/runtime_libart.mk
PRODUCT_SYSTEM_PROPERTIES += \
    ro.dalvik.vm.native.bridge=libberberis.so

PRODUCT_SYSTEM_PROPERTIES += \
    ro.dalvik.vm.isa.riscv64=x86_64 \
    ro.enable.native.bridge.exec=1

PRODUCT_SOONG_NAMESPACES += frameworks/libs/native_bridge_support/libc

BUILD_BERBERIS := true
