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

#include "berberis/guest_loader/guest_loader.h"

#include <mutex>
#include <thread>

#include "berberis/base/config_globals.h"  // SetMainExecutableRealPath
#include "berberis/base/stringprintf.h"
#include "berberis/base/tracing.h"
#include "berberis/guest_abi/guest_params.h"
#include "berberis/guest_os_primitives/guest_thread.h"
#include "berberis/guest_state/guest_addr.h"
#include "berberis/guest_state/guest_state.h"
#include "berberis/kernel_api/sys_mman_emulation.h"
#include "berberis/runtime_primitives/host_call_frame.h"
#include "berberis/runtime_primitives/host_function_wrapper_impl.h"  // MakeTrampolineCallable
#include "berberis/tiny_loader/tiny_loader.h"
#include "native_bridge_support/linker/static_tls_config.h"

#include "private/CFIShadow.h"  // kLibraryAlignment

#include "app_process.h"
#include "guest_loader_impl.h"

namespace berberis {

namespace {

const char* FindPtInterp(const LoadedElfFile* loaded_executable) {
  const ElfPhdr* phdr_table = loaded_executable->phdr_table();
  size_t phdr_count = loaded_executable->phdr_count();
  ElfAddr load_bias = loaded_executable->load_bias();

  for (size_t i = 0; i < phdr_count; ++i) {
    const auto& phdr = phdr_table[i];
    if (phdr.p_type == PT_INTERP) {
      return reinterpret_cast<const char*>(load_bias + phdr.p_vaddr);
    }
  }

  return nullptr;
}

// Never returns.
void StartGuestExecutableImpl(size_t argc,
                              const char* argv[],
                              char* envp[],
                              const LoadedElfFile* linker_elf_file,
                              const LoadedElfFile* main_executable_elf_file,
                              const LoadedElfFile* vdso_elf_file) {
  GuestAddr main_executable_entry_point = ToGuestAddr(main_executable_elf_file->entry_point());
  GuestAddr entry_point = linker_elf_file->is_loaded() ? ToGuestAddr(linker_elf_file->entry_point())
                                                       : main_executable_entry_point;

  // Passing environ to guest "as is" is ok unless/until proven otherwise.
  StartExecutable(argc,
                  argv,
                  envp,
                  ToGuestAddr(linker_elf_file->base_addr()),
                  entry_point,
                  main_executable_entry_point,
                  ToGuestAddr(main_executable_elf_file->phdr_table()),
                  main_executable_elf_file->phdr_count(),
                  ToGuestAddr(vdso_elf_file->base_addr()));
}

// ATTENTION: Assume guest and host integer and pointer types match.
class FormatBufferGuestParamsArgs {
 public:
  // Capture ephemeral GuestVAListParams into internal params_ variable.
  // GuestVAListParams is ephemeral in most cases because it's either produced from GuestParams or
  // from std::va_list argument.
  explicit FormatBufferGuestParamsArgs(GuestVAListParams&& params) : params_(params) {}

  const char* GetCStr() { return params_.GetPointerParam<const char>(); }
  uintmax_t GetPtrAsUInt() { return params_.GetParam<GuestAddr>(); }
  intmax_t GetInt() { return params_.GetParam<int>(); }
  intmax_t GetLong() { return params_.GetParam<long>(); }
  intmax_t GetLongLong() { return params_.GetParam<long long>(); }
  uintmax_t GetUInt() { return params_.GetParam<unsigned int>(); }
  uintmax_t GetULong() { return params_.GetParam<unsigned long>(); }
  uintmax_t GetULongLong() { return params_.GetParam<unsigned long long>(); }
  intmax_t GetChar() { return params_.GetParam<int>(); }
  uintmax_t GetSizeT() { return params_.GetParam<GuestAddr>(); }

 private:
  GuestVAListParams params_;
};

void TraceCallback(HostCode callee, ThreadState* state) {
  UNUSED(callee);

  if (Tracing::IsOn()) {
    auto [format] = GuestParamsValues<void(const char*, ...)>(state);
    FormatBufferGuestParamsArgs args{GuestParamsValues<void(const char*, ...)>(state)};
    Tracing::TraceA(format, &args);
  }
}

void PostInitCallback(HostCode callee, ThreadState* state) {
  UNUSED(callee, state);

  AppProcessPostInit();
}

void InterceptGuestSymbolCallback(HostCode callee, ThreadState* state) {
  UNUSED(callee, state);

  // TODO(b/284194965): Uncomment when proxy_loader is ready.
  // auto [addr, lib_name, sym_name] = GuestParamsValues<decltype(InterceptGuestSymbol)>(state);
  // InterceptGuestSymbol(addr, lib_name, sym_name);
}

void ConfigStaticTlsCallback(HostCode callee, ThreadState* state) {
  UNUSED(callee);

  auto [config] = GuestParamsValues<void(const NativeBridgeStaticTlsConfig*)>(state);
  state->thread->ConfigStaticTls(config);
}

void GetHostPthreadCallback(HostCode callee, ThreadState* state) {
  UNUSED(callee);

  auto&& [ret] = GuestReturnReference<decltype(pthread_self)>(state);
  ret = pthread_self();
}

bool InitializeVdso(const LoadedElfFile& vdso_elf_file, std::string* error_msg) {
  if (!MakeElfSymbolTrampolineCallable(
          vdso_elf_file, "vdso", "native_bridge_trace", TraceCallback, nullptr, error_msg)) {
    return false;
  }

  if (!MakeElfSymbolTrampolineCallable(vdso_elf_file,
                                       "vdso",
                                       "native_bridge_intercept_symbol",
                                       InterceptGuestSymbolCallback,
                                       nullptr,
                                       error_msg)) {
    return false;
  }

  if (!MakeElfSymbolTrampolineCallable(
          vdso_elf_file, "vdso", "native_bridge_post_init", PostInitCallback, nullptr, error_msg)) {
    return false;
  }

  void* call_guest_addr = vdso_elf_file.FindSymbol("native_bridge_call_guest");
  if (call_guest_addr == nullptr) {
    *error_msg = "couldn't find \"native_bridge_call_guest\" symbol in vdso";
    return false;
  }
  InitHostCallFrameGuestPC(ToGuestAddr(call_guest_addr));

  return true;
}

bool InitializeLinker(LinkerCallbacks* linker_callbacks,
                      const LoadedElfFile& linker_elf_file,
                      std::string* error_msg) {
  if (!MakeElfSymbolTrampolineCallable(linker_elf_file,
                                       "linker",
                                       "__native_bridge_config_static_tls",
                                       ConfigStaticTlsCallback,
                                       nullptr,
                                       error_msg)) {
    return false;
  }

  if (!MakeElfSymbolTrampolineCallable(linker_elf_file,
                                       "linker",
                                       "__native_bridge_get_host_pthread",
                                       GetHostPthreadCallback,
                                       nullptr,
                                       error_msg)) {
    return false;
  }

  return InitializeLinkerCallbacks(linker_callbacks, linker_elf_file, error_msg) &&
         InitializeLinkerCallbacksArch(linker_callbacks, linker_elf_file, error_msg);
}

std::mutex g_guest_loader_instance_mtx;
GuestLoader* g_guest_loader_instance;

}  // namespace

GuestLoader::GuestLoader() = default;

GuestLoader::~GuestLoader() = default;

GuestLoader* GuestLoader::CreateInstance(const char* main_executable_path,
                                         const char* vdso_path,
                                         const char* loader_path,
                                         std::string* error_msg) {
  std::lock_guard<std::mutex> lock(g_guest_loader_instance_mtx);
  CHECK(g_guest_loader_instance == nullptr);

  TRACE(
      "GuestLoader::CreateInstance(main_executable_path=\"%s\", "
      "vdso_path=\"%s\", loader_path=\"%s\")",
      main_executable_path,
      vdso_path,
      loader_path);

  std::unique_ptr<GuestLoader> instance(new GuestLoader());

  if (!TinyLoader::LoadFromFile(main_executable_path,
                                kLibraryAlignment,
                                &MmapForGuest,
                                &MunmapForGuest,
                                &instance->executable_elf_file_,
                                error_msg)) {
    return nullptr;
  }

  // For readlink(/proc/self/exe).
  SetMainExecutableRealPath(main_executable_path);

  instance->main_executable_path_ = main_executable_path;
  // Initialize caller_addr_ to executable entry point.
  instance->caller_addr_ = instance->executable_elf_file_.entry_point();

  // Real pt_interp is only used to distinguish static executables.
  bool is_static_executable = (FindPtInterp(&instance->executable_elf_file_) == nullptr);

  if (TinyLoader::LoadFromFile(vdso_path,
                               kLibraryAlignment,
                               &MmapForGuest,
                               &MunmapForGuest,
                               &instance->vdso_elf_file_,
                               error_msg)) {
    if (!InitializeVdso(instance->vdso_elf_file_, error_msg)) {
      return nullptr;
    }
  } else {
    if (!is_static_executable) {
      return nullptr;
    }
  }

  if (is_static_executable) {
    InitializeLinkerCallbacksToStubs(&instance->linker_callbacks_);
    if (instance->executable_elf_file_.e_type() == ET_DYN) {
      // Special case - ET_DYN executable without PT_INTERP, consider linker.
      TRACE("pretend running linker as main executable");
      if (!InitializeLinker(
              &instance->linker_callbacks_, instance->executable_elf_file_, error_msg)) {
        // Not the right linker, warn and hope for the best.
        TRACE("failed to init main executable as linker, running as is");
      }
    }
  } else {
    if (!TinyLoader::LoadFromFile(loader_path,
                                  kLibraryAlignment,
                                  &MmapForGuest,
                                  &MunmapForGuest,
                                  &instance->linker_elf_file_,
                                  error_msg)) {
      return nullptr;
    }
    if (!InitializeLinker(&instance->linker_callbacks_, instance->linker_elf_file_, error_msg)) {
      return nullptr;
    }
    InitLinkerDebug(instance->linker_elf_file_);
  }

  g_guest_loader_instance = instance.release();
  return g_guest_loader_instance;
}

GuestLoader* GuestLoader::GetInstance() {
  std::lock_guard<std::mutex> lock(g_guest_loader_instance_mtx);
  CHECK(g_guest_loader_instance != nullptr);
  return g_guest_loader_instance;
}

void GuestLoader::StartGuestMainThread() {
  std::thread t(StartGuestExecutableImpl,
                1,
                &main_executable_path_,
                environ,
                &linker_elf_file_,
                &executable_elf_file_,
                &vdso_elf_file_);
  t.detach();
  WaitForAppProcess();
}

void GuestLoader::StartGuestExecutable(size_t argc, const char* argv[], char* envp[]) {
  StartGuestExecutableImpl(
      argc, argv, envp, &linker_elf_file_, &executable_elf_file_, &vdso_elf_file_);
}

GuestLoader* GuestLoader::StartAppProcessInNewThread(std::string* error_msg) {
  GuestLoader* instance = CreateInstance(kAppProcessPath, kVdsoPath, kPtInterpPath, error_msg);
  if (instance) {
    instance->StartGuestMainThread();
  }
  return instance;
}

void GuestLoader::StartExecutable(const char* main_executable_path,
                                  const char* vdso_path,
                                  const char* loader_path,
                                  size_t argc,
                                  const char* argv[],
                                  char* envp[],
                                  std::string* error_msg) {
  GuestLoader* instance = CreateInstance(main_executable_path,
                                         vdso_path ? vdso_path : kVdsoPath,
                                         loader_path ? loader_path : kPtInterpPath,
                                         error_msg);
  if (instance) {
    instance->StartGuestExecutable(argc, argv, envp);
  }
}

const struct r_debug* GuestLoader::FindRDebug() const {
  if (executable_elf_file_.is_loaded() && executable_elf_file_.dynamic() != nullptr) {
    for (const ElfDyn* d = executable_elf_file_.dynamic(); d->d_tag != DT_NULL; ++d) {
      if (d->d_tag == DT_DEBUG) {
        return reinterpret_cast<const struct r_debug*>(d->d_un.d_val);
      }
    }
  }

  return nullptr;
}

bool MakeElfSymbolTrampolineCallable(const LoadedElfFile& elf_file,
                                     const char* elf_file_label,
                                     const char* symbol_name,
                                     void (*callback)(HostCode, ThreadState*),
                                     HostCode arg,
                                     std::string* error_msg) {
  void* symbol_addr = elf_file.FindSymbol(symbol_name);
  if (symbol_addr == nullptr) {
    *error_msg = StringPrintf("couldn't find \"%s\" symbol in %s", symbol_name, elf_file_label);
    return false;
  }
  MakeTrampolineCallable(ToGuestAddr(symbol_addr), false, callback, arg, symbol_name);
  return true;
}

}  // namespace berberis