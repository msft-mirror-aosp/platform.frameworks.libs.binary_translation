/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include "native_bridge.h"

#include <dlfcn.h>
#include <libgen.h>
#include <stdio.h>
#include <sys/system_properties.h>

#include <string_view>

#include "procinfo/process_map.h"

#include "berberis/base/bit_util.h"
#include "berberis/base/config_globals.h"
#include "berberis/base/logging.h"
#include "berberis/base/strings.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_abi/guest_call.h"
#include "berberis/guest_loader/guest_loader.h"
#include "berberis/guest_os_primitives/guest_map_shadow.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/jni/jni_trampolines.h"
#include "berberis/native_activity/native_activity_wrapper.h"
#include "berberis/native_bridge/native_bridge.h"
#include "berberis/runtime/berberis.h"
#include "berberis/runtime_primitives/known_guest_function_wrapper.h"

#define LOG_NB ALOGV  // redefine to ALOGD for debugging

namespace {

// See android/system/core/libnativebridge/native_bridge.cc
// Even thought berberis does not support early version of NB interface
// (deprecated methods do not work anymore) v2 support is needed to have NB call
// getSignalHandler function.
const constexpr uint32_t kNativeBridgeCallbackMinVersion = 2;
const constexpr uint32_t kNativeBridgeCallbackVersion = 7;
const constexpr uint32_t kNativeBridgeCallbackMaxVersion = kNativeBridgeCallbackVersion;

const android::NativeBridgeRuntimeCallbacks* g_runtime_callbacks = nullptr;

// Treble uses "sphal" name for the vendor namespace.
constexpr const char* kVendorNamespaceName = "sphal";

android::native_bridge_namespace_t* ToNbNamespace(android_namespace_t* android_ns) {
  return reinterpret_cast<android::native_bridge_namespace_t*>(android_ns);
}

android_namespace_t* ToAndroidNamespace(android::native_bridge_namespace_t* nb_ns) {
  return reinterpret_cast<android_namespace_t*>(nb_ns);
}

class NdktNativeBridge {
 public:
  NdktNativeBridge();
  ~NdktNativeBridge();

  bool Initialize(std::string* error_msg);
  void* LoadLibrary(const char* libpath, int flags);
  void* LoadLibrary(const char* libpath, int flags, const android_dlextinfo* extinfo);
  berberis::GuestAddr DlSym(void* handle, const char* name);
  const char* DlError();
  bool InitAnonymousNamespace(const char* public_ns_sonames, const char* anon_ns_library_path);
  android::native_bridge_namespace_t* CreateNamespace(
      const char* name,
      const char* ld_library_path,
      const char* default_library_path,
      uint64_t type,
      const char* permitted_when_isolated_path,
      android::native_bridge_namespace_t* parent_ns);
  android::native_bridge_namespace_t* GetExportedNamespace(const char* name);
  bool LinkNamespaces(android::native_bridge_namespace_t* from,
                      android::native_bridge_namespace_t* to,
                      const char* shared_libs_sonames);

 private:
  bool FinalizeInit();

  berberis::GuestLoader* guest_loader_;
};

NdktNativeBridge::NdktNativeBridge() : guest_loader_(nullptr) {}

NdktNativeBridge::~NdktNativeBridge() {}

bool NdktNativeBridge::Initialize(std::string* error_msg) {
  guest_loader_ = berberis::GuestLoader::StartAppProcessInNewThread(error_msg);
  berberis::RegisterKnownGuestFunctionWrapper("JNI_OnLoad", berberis::WrapGuestJNIOnLoad);
  berberis::RegisterKnownGuestFunctionWrapper("ANativeActivity_onCreate",
                                              berberis::WrapGuestNativeActivityOnCreate);
  return guest_loader_ != nullptr;
}

void* NdktNativeBridge::LoadLibrary(const char* libpath, int flags) {
  return LoadLibrary(libpath, flags, nullptr);
}

void* NdktNativeBridge::LoadLibrary(const char* libpath,
                                    int flags,
                                    const android_dlextinfo* extinfo) {
  // We don't have a callback after all java initialization is finished. So we call the finalizing
  // routine from here, just before we load any app's native code.
  static bool init_finalized = FinalizeInit();
  UNUSED(init_finalized);
  return guest_loader_->DlOpenExt(libpath, flags, extinfo);
}

berberis::GuestAddr NdktNativeBridge::DlSym(void* handle, const char* name) {
  return guest_loader_->DlSym(handle, name);
}

const char* NdktNativeBridge::DlError() {
  return guest_loader_->DlError();
}

android::native_bridge_namespace_t* NdktNativeBridge::CreateNamespace(
    const char* name,
    const char* ld_library_path,
    const char* default_library_path,
    uint64_t type,
    const char* permitted_when_isolated_path,
    android::native_bridge_namespace_t* parent_ns) {
  return ToNbNamespace(guest_loader_->CreateNamespace(name,
                                                      ld_library_path,
                                                      default_library_path,
                                                      type,
                                                      permitted_when_isolated_path,
                                                      ToAndroidNamespace(parent_ns)));
}

android::native_bridge_namespace_t* NdktNativeBridge::GetExportedNamespace(const char* name) {
  return ToNbNamespace(guest_loader_->GetExportedNamespace(name));
}

bool NdktNativeBridge::InitAnonymousNamespace(const char* public_ns_sonames,
                                              const char* anon_ns_library_path) {
  return guest_loader_->InitAnonymousNamespace(public_ns_sonames, anon_ns_library_path);
}

bool NdktNativeBridge::LinkNamespaces(android::native_bridge_namespace_t* from,
                                      android::native_bridge_namespace_t* to,
                                      const char* shared_libs_sonames) {
  return guest_loader_->LinkNamespaces(
      ToAndroidNamespace(from), ToAndroidNamespace(to), shared_libs_sonames);
}

void ProtectMappingsFromGuest() {
  auto callback =
      [](uint64_t start, uint64_t end, uint16_t, uint64_t, ino_t, const char* libname_c_str, bool) {
        std::string_view libname(libname_c_str);
        // Per analysis in b/218772975 only libc is affected. It's occasionally either proxy libc or
        // guest libc. So we protect all libs with "libc.so" substring. At this point no app's libs
        // are loaded yet, so the app shouldn't tamper with the already loaded ones. We don't
        // protect all the already loaded libraries though since GuestMapShadow isn't optimized
        // to work with large number of entries. Also some of them could be unmapped later, which is
        // not expected for libc.so.
        if (libname.find("libc.so") != std::string_view::npos) {
          berberis::GuestMapShadow::GetInstance()->AddProtectedMapping(
              berberis::bit_cast<void*>(static_cast<uintptr_t>(start)),
              berberis::bit_cast<void*>(static_cast<uintptr_t>(end)));
        }
      };
  android::procinfo::ReadMapFile("/proc/self/maps", callback);
}

extern "C" const char* __progname;

bool NdktNativeBridge::FinalizeInit() {
  // Guest-libc is expected to be loaded along with app-process during Initialize(). At that time
  // __progname isn't yet initialized in java. So now when it should be initialized we copy it over
  // from host to guest.
  // Note that we cannot delay Initialize() (and hence guest-libc loading) until now because
  // guest_loader initialized there is then used to create and link linker namespaces.
  // We cannot cannot unload (dlclose) guest-libc after app-process loading either (intending to
  // reload it now to get the updated __progname), since guest_linker is already tightly linked with
  // it.

  // Force libc loading if it's not loaded yet to ensure the symbol is overridden.
  // We do not call LoadLibrary since it'd recurse back into FinalizeInit.
  void* libc = guest_loader_->DlOpenExt("libc.so", RTLD_NOW, nullptr);
  CHECK_NE(libc, nullptr);
  auto addr = DlSym(libc, "__progname");
  CHECK_NE(addr, berberis::kNullGuestAddr);
  memcpy(berberis::ToHostAddr<char*>(addr), &__progname, sizeof(__progname));

  // Now, when guest libc and proxy-libc are loaded,
  // remember mappings which guest code must not tamper with.
  ProtectMappingsFromGuest();

  return true;
}

NdktNativeBridge g_ndkt_native_bridge;

// Runtime values must be non-NULL, otherwise native bridge will be disabled.
// Note, that 'supported_abis' and 'abi_count' are deprecated (b/18061712).
const struct android::NativeBridgeRuntimeValues* GetAppEnvByIsa(const char* app_isa) {
  if (app_isa == nullptr) {
    ALOGE("instruction set is null");
    return nullptr;
  }

  if (strcmp(app_isa, berberis::kGuestIsa) == 0) {
    return &berberis::kNativeBridgeRuntimeValues;
  }

  ALOGE("unknown instruction set '%s'", app_isa);
  return nullptr;
}

void SetAppPropertiesFromCodeCachePath(const char* private_dir) {
  if (private_dir == nullptr) {
    return;
  }

  // Expect private_dir to be .../<app_package>/code_cache
  std::string_view path(private_dir);
  if (!berberis::ConsumeSuffix(&path, "/code_cache")) {
    return;
  }

  berberis::SetAppPrivateDir(path);

  auto begin = path.find_last_of('/');
  if (begin == std::string_view::npos) {
    return;
  }
  berberis::SetAppPackageName(path.substr(begin + 1));
}

bool native_bridge_initialize(const android::NativeBridgeRuntimeCallbacks* runtime_cbs,
                              const char* private_dir,
                              const char* instruction_set) {
  LOG_NB("native_bridge_initialize(runtime_callbacks=%p, private_dir='%s', app_isa='%s')",
         runtime_cbs,
         private_dir ? private_dir : "(null)",
         instruction_set ? instruction_set : "(null)");
  auto* env = GetAppEnvByIsa(instruction_set);
  if (env == nullptr) {
    return false;
  }
  g_runtime_callbacks = runtime_cbs;
  SetAppPropertiesFromCodeCachePath(private_dir);
  berberis::InitBerberis();

  char version[PROP_VALUE_MAX];
  if (__system_property_get("ro.berberis.version", version)) {
    ALOGI("Initialized Berberis (%s), version %s", env->os_arch, version);
  } else {
    ALOGI("Initialized Berberis (%s)", env->os_arch);
  }

  std::string error_msg;
  if (!g_ndkt_native_bridge.Initialize(&error_msg)) {
    LOG_ALWAYS_FATAL("native_bridge_initialize: %s", error_msg.c_str());
  }
  return true;
}

void* native_bridge_loadLibrary(const char* libpath, int flag) {
  // We should only get here if this library is not native.
  LOG_NB("native_bridge_loadLibrary(path='%s', flag=0x%x)", libpath ? libpath : "(null)", flag);

  return g_ndkt_native_bridge.LoadLibrary(libpath, flag);
}

void* native_bridge_getTrampolineWithJNICallType(void* handle,
                                                 const char* name,
                                                 const char* shorty,
                                                 uint32_t len,
                                                 enum android::JNICallType jni_call_type) {
  LOG_NB(
      "native_bridge_getTrampolineWithJNICallType(handle=%p, name='%s', shorty='%s', len=%d, "
      "jni_call_type=%d)",
      handle,
      name ? name : "(null)",
      shorty ? shorty : "(null)",
      len,
      jni_call_type);

  berberis::GuestAddr guest_addr = g_ndkt_native_bridge.DlSym(handle, name);
  if (!guest_addr) {
    return nullptr;
  }

  if (shorty) {
    return const_cast<void*>(berberis::WrapGuestJNIFunction(
        guest_addr,
        shorty,
        name,
        jni_call_type != android::JNICallType::kJNICallTypeCriticalNative));
  }

  berberis::HostCode result = berberis::WrapKnownGuestFunction(guest_addr, name);
  if (result == nullptr) {
    // No wrapper is registered for this function name.
    // This usually happens for ANativeActivity_onCreate renamed with android.app.func_name.
    // TODO(b/27307664): maybe query android.app.func_name from Java and check exactly?
    TRACE("No wrapper is registered for %s, assume it's ANativeActivity_onCreate", name);
    result = berberis::WrapKnownGuestFunction(guest_addr, "ANativeActivity_onCreate");
  }
  return const_cast<void*>(result);
}

void* native_bridge_getTrampoline(void* handle,
                                  const char* name,
                                  const char* shorty,
                                  uint32_t len) {
  LOG_NB(
      "Warning: Unexpected call to native_bridge_getTrampoline (old android version?), converting "
      "to a native_bridge_getTrampolineWithJNICallType call with kJNICallTyepRegular");
  return native_bridge_getTrampolineWithJNICallType(
      handle, name, shorty, len, android::JNICallType::kJNICallTypeRegular);
}

bool native_bridge_isSupported(const char* libpath) {
  LOG_NB("native_bridge_isSupported(path='%s')", libpath ? libpath : "(null)");
  return true;
}

const struct android::NativeBridgeRuntimeValues* native_bridge_getAppEnv(
    const char* instruction_set) {
  LOG_NB("native_bridge_getAppEnv(app_isa='%s')", instruction_set ? instruction_set : "(null)");
  return GetAppEnvByIsa(instruction_set);
}

bool native_bridge_isCompatibleWith(uint32_t bridge_version) {
  LOG_NB("native_bridge_isCompatibleWith(bridge_version=%d)", bridge_version);
  return bridge_version >= kNativeBridgeCallbackMinVersion &&
         bridge_version <= kNativeBridgeCallbackMaxVersion;
}

android::NativeBridgeSignalHandlerFn native_bridge_getSignalHandler(int signal) {
  LOG_NB("native_bridge_getSignalHandler(signal=%d)", signal);
  return nullptr;
}

int native_bridge_unloadLibrary(void* handle) {
  LOG_NB("native_bridge_unloadLibrary(handle=%p)", handle);
  // TODO(b/276787500): support library unloading!
  return 0;
}

const char* native_bridge_getError() {
  LOG_NB("native_bridge_getError()");
  return g_ndkt_native_bridge.DlError();
}

bool native_bridge_isPathSupported(const char* library_path) {
  LOG_NB("native_bridge_isPathSupported(path=%s)", library_path);
  return strstr(library_path, berberis::kSupportedLibraryPathSubstring) != nullptr;
}

bool native_bridge_initAnonymousNamespace(const char* public_ns_sonames,
                                          const char* anon_ns_library_path) {
  LOG_NB("native_bridge_initAnonymousNamespace(public_ns_sonames=%s, anon_ns_library_path=%s)",
         public_ns_sonames,
         anon_ns_library_path);
  return g_ndkt_native_bridge.InitAnonymousNamespace(public_ns_sonames, anon_ns_library_path);
}

android::native_bridge_namespace_t* native_bridge_createNamespace(
    const char* name,
    const char* ld_library_path,
    const char* default_library_path,
    uint64_t type,
    const char* permitted_when_isolated_path,
    android::native_bridge_namespace_t* parent_ns) {
  LOG_NB("native_bridge_createNamespace(name=%s, path=%s)", name, ld_library_path);
  return g_ndkt_native_bridge.CreateNamespace(
      name, ld_library_path, default_library_path, type, permitted_when_isolated_path, parent_ns);
}

bool native_bridge_linkNamespaces(android::native_bridge_namespace_t* from,
                                  android::native_bridge_namespace_t* to,
                                  const char* shared_libs_sonames) {
  LOG_NB("native_bridge_linkNamespaces(from=%p, to=%p, shared_libs=%s)",
         from,
         to,
         shared_libs_sonames);

  return g_ndkt_native_bridge.LinkNamespaces(from, to, shared_libs_sonames);
}

void* native_bridge_loadLibraryExt(const char* libpath,
                                   int flag,
                                   android::native_bridge_namespace_t* ns) {
  LOG_NB("native_bridge_loadLibraryExt(path=%s)", libpath);

  android_dlextinfo extinfo;
  extinfo.flags = ANDROID_DLEXT_USE_NAMESPACE;
  extinfo.library_namespace = ToAndroidNamespace(ns);

  return g_ndkt_native_bridge.LoadLibrary(libpath, flag, &extinfo);
}

android::native_bridge_namespace_t* native_bridge_getVendorNamespace() {
  LOG_NB("native_bridge_getVendorNamespace()");
  // This method is retained for backwards compatibility.
  return g_ndkt_native_bridge.GetExportedNamespace(kVendorNamespaceName);
}

android::native_bridge_namespace_t* native_bridge_getExportedNamespace(const char* name) {
  LOG_NB("native_bridge_getExportedNamespace(name=%s)", name);
  return g_ndkt_native_bridge.GetExportedNamespace(name);
}

void native_bridge_preZygoteFork() {
  // In case of app-zygote the translator could have executed some guest code
  // during app-zygote's doPreload(). Zygote's fork doesn't allow unrecognized
  // open file descriptors, so we close them.
  //
  // We assume that all guest execution has finished in doPreload() and there
  // are no background guest threads. ART ensures the fork is single-threaded by
  // calling waitUntilAllThreadsStopped() in ZygoteHooks::preZork().
  // TODO(b/188923523): Technically this happens after nativePreFork() (which
  // calls this callback), so theoretically some guest thread may still be
  // running and finishes later. If this happens to be an issue, we can call an
  // analog of waitUntilAllThreadsStopped() here. Or try to call nativePreFork()
  // after waitUntilAllThreadsStopped() in ART.

  // TODO(b/188923523): Consider moving to berberis::GuestPreZygoteFork().
  void* liblog = g_ndkt_native_bridge.LoadLibrary("liblog.so", RTLD_NOLOAD);
  // Nothing to close if the library hasn't been loaded.
  if (liblog) {
    auto addr = g_ndkt_native_bridge.DlSym(liblog, "__android_log_close");
    CHECK_NE(addr, berberis::kNullGuestAddr);
    berberis::GuestCall call;
    call.RunVoid(addr);
  }

  berberis::PreZygoteForkUnsafe();
}

}  // namespace

namespace berberis {

const char* GetJMethodShorty(JNIEnv* env, jmethodID mid) {
  CHECK(g_runtime_callbacks);
  return (g_runtime_callbacks->getMethodShorty)(env, mid);
}

}  // namespace berberis

extern "C" {
// "NativeBridgeItf" is effectively an API (it is the name of the symbol that
// will be loaded by the native bridge library).
android::NativeBridgeCallbacks NativeBridgeItf = {
    kNativeBridgeCallbackVersion,
    &native_bridge_initialize,
    &native_bridge_loadLibrary,
    &native_bridge_getTrampoline,
    &native_bridge_isSupported,
    &native_bridge_getAppEnv,
    &native_bridge_isCompatibleWith,
    &native_bridge_getSignalHandler,
    &native_bridge_unloadLibrary,
    &native_bridge_getError,
    &native_bridge_isPathSupported,
    &native_bridge_initAnonymousNamespace,
    &native_bridge_createNamespace,
    &native_bridge_linkNamespaces,
    &native_bridge_loadLibraryExt,
    &native_bridge_getVendorNamespace,
    &native_bridge_getExportedNamespace,
    &native_bridge_preZygoteFork,
    &native_bridge_getTrampolineWithJNICallType,
};
}  // extern "C"
