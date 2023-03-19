/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "berberis/base/config_globals.h"

#include <cstdlib>
#include <cstring>

#if defined(__BIONIC__)
#include <sys/system_properties.h>
#endif

#include <bitset>
#include <string_view>

#include "berberis/base/checks.h"
#include "berberis/base/forever_alloc.h"
#include "berberis/base/logging.h"
#include "berberis/base/strings.h"

namespace berberis {

namespace {

const char* g_main_executable_real_path = nullptr;

const char* g_app_package_name = nullptr;
const char* g_app_private_dir = nullptr;

const char* MakeForeverCStr(std::string_view view) {
  auto str = reinterpret_cast<char*>(AllocateForever(view.size() + 1, alignof(char)));
  memcpy(str, view.data(), view.size());
  str[view.size()] = '\0';
  return str;
}

#if defined(__BIONIC__)

bool TryReadBionicSystemPropertyImpl(const std::string_view prop_name, const char** value_ptr) {
  auto pi = __system_property_find(prop_name.data());
  if (!pi) {
    return false;
  }
  __system_property_read_callback(
      pi,
      [](void* cookie, const char*, const char* value, unsigned) {
        *reinterpret_cast<const char**>(cookie) = MakeForeverCStr(value);
      },
      value_ptr);
  return true;
}

bool TryReadBionicSystemProperty(const std::string_view prop_name, const char** value_ptr) {
  // Allow properties without ro. prefix (they override ro. properties).
  if (prop_name.size() > 3 && prop_name.substr(0, 3) == "ro." &&
      TryReadBionicSystemPropertyImpl(prop_name.substr(3), value_ptr)) {
    return true;
  }

  return TryReadBionicSystemPropertyImpl(prop_name, value_ptr);
}

#endif

bool TryReadConfig(const char* env_name,
                   [[maybe_unused]] const std::string_view prop_name,
                   const char** value_ptr) {
  if (auto env = getenv(env_name)) {
    *value_ptr = MakeForeverCStr(env);
    return true;
  }
#if defined(__BIONIC__)
  return TryReadBionicSystemProperty(prop_name, value_ptr);
#else
  return false;
#endif
}

class ConfigStr {
 public:
  ConfigStr(const char* env_name, [[maybe_unused]] const char* prop_name) {
    TryReadConfig(env_name, prop_name, &value_);
  }

  [[nodiscard]] const char* get() const { return value_; }

 private:
  const char* value_ = nullptr;
};

std::string ToString(ConfigFlag flag) {
  switch (flag) {
    case kVerboseTranslation:
      return "verbose-translation";
    case kNumConfigFlags:
      break;
  }
  return "<unknown-config-flag>";
}

std::bitset<kNumConfigFlags> MakeConfigFlagsSet() {
  ConfigStr var("BERBERIS_FLAGS", "ro.berberis.flags");
  std::bitset<kNumConfigFlags> flags_set;
  if (!var.get()) {
    return flags_set;
  }
  auto token_vector = Split(var.get(), ",");
  for (const auto& token : token_vector) {
    bool found = false;
    for (int flag = 0; flag < kNumConfigFlags; flag++) {
      if (token == ToString(ConfigFlag(flag))) {
        flags_set.set(flag);
        found = true;
        break;
      }
    }
    if (!found) {
      ALOGW("Unrecognized config flag '%s' - ignoring", token.c_str());
    }
  }
  return flags_set;
}

}  // namespace

void SetMainExecutableRealPath(std::string_view path) {
  CHECK(!path.empty());
  CHECK_EQ('/', path[0]);
  g_main_executable_real_path = MakeForeverCStr(path);
}

const char* GetMainExecutableRealPath() {
  return g_main_executable_real_path;
}

void SetAppPackageName(std::string_view name) {
  CHECK(!name.empty());
  g_app_package_name = MakeForeverCStr(name);
}

const char* GetAppPackageName() {
  return g_app_package_name;
}

void SetAppPrivateDir(std::string_view name) {
  CHECK(!name.empty());
  g_app_private_dir = MakeForeverCStr(name);
}

const char* GetAppPrivateDir() {
  return g_app_private_dir;
}

const char* GetTracingConfig() {
  static ConfigStr var("BERBERIS_TRACING", "berberis.tracing");
  return var.get();
}

const char* GetTranslationModeConfig() {
  static ConfigStr var("BERBERIS_MODE", "berberis.mode");
  return var.get();
}

const char* GetProfilingConfig() {
  static ConfigStr var("BERBERIS_PROFILING", "berberis.profiling");
  return var.get();
}

bool IsConfigFlagSet(ConfigFlag flag) {
  static auto flags_set = MakeConfigFlagsSet();
  return flags_set.test(flag);
}

}  // namespace berberis
