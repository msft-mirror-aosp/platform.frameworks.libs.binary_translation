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

include frameworks/libs/native_bridge_support/native_bridge_support.mk

BERBERIS_PRODUCT_PACKAGES := \
    berberis_program_runner_riscv64 \
    libberberis_exec_region

# TODO(b/277625560): Include $(NATIVE_BRIDGE_PRODUCT_PACKAGES) instead
# when all its bits are ready for riscv64.
BERBERIS_PRODUCT_PACKAGES += $(NATIVE_BRIDGE_PRODUCT_PACKAGES_RISCV64_READY)


BERBERIS_DEV_PRODUCT_PACKAGES := \
    berberis_dwarf_reader \
    berberis_host_tests \
    berberis_nogrod_unit_tests
