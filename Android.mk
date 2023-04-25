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

LOCAL_PATH := $(call my-dir)

# Berberis includes some components which may conflict with other packages.
# Only build it when requested explicitly.
ifeq ($(BUILD_BERBERIS),true)

include $(LOCAL_PATH)/riscv64_to_x86_64_config.mk

.PHONY: berberis_all
berberis_all: \
    $(BERBERIS_PRODUCT_PACKAGES) \
    $(BERBERIS_DEV_PRODUCT_PACKAGES)

endif  # BUILD_BERBERIS
