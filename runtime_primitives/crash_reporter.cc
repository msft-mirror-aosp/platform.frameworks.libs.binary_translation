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

#include "berberis/runtime_primitives/crash_reporter.h"

#include <sys/syscall.h>  // SYS_rt_tgdigqueueinfo
#include <unistd.h>       // syscall

#include <csignal>

#include "berberis/base/gettid.h"
#include "berberis/base/tracing.h"
#include "berberis/instrument/crash.h"

namespace berberis {

namespace {

struct sigaction g_orig_action[NSIG];

}  // namespace

void HandleFatalSignal(int sig, siginfo_t* info, void* context) {
  TRACE("Fatal signal %d", sig);

  OnCrash(sig, info, context);

  // Let the default crash reporter do the job. Restore the original signal action, as the default
  // crash reporter can re-raise the signal.
  sigaction(sig, &g_orig_action[sig], nullptr);
  if (g_orig_action[sig].sa_flags & SA_SIGINFO) {
    // Run the original signal action manually and provide actual siginfo and context.
    g_orig_action[sig].sa_sigaction(sig, info, context);
  } else {
    // This should be rare as debuggerd sets siginfo handlers for most signals. The original action
    // doesn't accept siginfo and context, so we re-raise the signal as accurate as possible and
    // hope for the best. If the signal is currently blocked we'll need to return from this handler
    // for the signal to be delivered.
    // TODO(b/232598137): Since the action doesn't accept siginfo it'll be ignored anyway, so
    // maybe we should just call g_orig_action[sig].sa_handler(sig) for immediate delivery.
    syscall(SYS_rt_tgsigqueueinfo, GetpidSyscall(), GettidSyscall(), sig, info);
  }
}

void InitCrashReporter() {
  struct sigaction action {};
  action.sa_sigaction = HandleFatalSignal;
  action.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigfillset(&action.sa_mask);

  sigaction(SIGSEGV, &action, &g_orig_action[SIGSEGV]);
  sigaction(SIGILL, &action, &g_orig_action[SIGILL]);
  sigaction(SIGFPE, &action, &g_orig_action[SIGFPE]);
  sigaction(SIGABRT, &action, &g_orig_action[SIGABRT]);
}

}  // namespace berberis
