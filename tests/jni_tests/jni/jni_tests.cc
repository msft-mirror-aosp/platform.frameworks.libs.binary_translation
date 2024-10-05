/*
 * Copyright (C) 2016 The Android Open Source Project
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

#include <jni.h>

namespace {

bool g_jni_onload_called = false;

jint add42(JNIEnv* /* env */, jclass /* clazz */, jint x) {
  return x + 42;
}

}  // namespace

extern "C" {

jint JNI_OnLoad(JavaVM* /* vm */, void* /* reserved */) {
  g_jni_onload_called = true;
  return JNI_VERSION_1_6;
}

JNIEXPORT jint JNICALL Java_com_berberis_jnitests_JniTests_intFromJNI(JNIEnv* /* env */,
                                                                      jclass /* clazz */) {
  return 42;
}

JNIEXPORT jboolean JNICALL
Java_com_berberis_jnitests_JniTests_isJNIOnLoadCalled(JNIEnv* /* env */, jclass /* clazz */) {
  return g_jni_onload_called;
}

JNIEXPORT jboolean JNICALL Java_com_berberis_jnitests_JniTests_checkGetVersion(JNIEnv* env,
                                                                               jclass /* clazz */) {
  return env->GetVersion() == JNI_VERSION_1_6;
}

JNIEXPORT jboolean JNICALL
Java_com_berberis_jnitests_JniTests_checkJavaVMCorrespondsToJNIEnv(JNIEnv* env,
                                                                   jclass /* clazz */) {
  JavaVM* vm;
  if (env->GetJavaVM(&vm) != JNI_OK) {
    return false;
  }

  void* env_copy;
  if (vm->GetEnv(&env_copy, JNI_VERSION_1_6) != JNI_OK) {
    return false;
  }

  return env == env_copy;
}

JNIEXPORT jboolean JNICALL Java_com_berberis_jnitests_JniTests_callRegisterNatives(JNIEnv* env,
                                                                                   jclass clazz) {
  JNINativeMethod methods[] = {{"add42", "(I)I", reinterpret_cast<void*>(&add42)}};
  return env->RegisterNatives(clazz, methods, sizeof(methods) / sizeof(methods[1])) == JNI_OK;
}

JNIEXPORT jint JNICALL Java_com_berberis_jnitests_JniTests_callAdd(JNIEnv* env,
                                                                   jclass clazz,
                                                                   jint x,
                                                                   jint y) {
  jmethodID method_id = env->GetStaticMethodID(clazz, "add", "(II)I");
  // ATTENTION: JNIEnv converts Call*Method(...) to Call*MethodV(va_list)!
  return env->CallStaticIntMethod(clazz, method_id, x, y);
}

JNIEXPORT jint JNICALL Java_com_berberis_jnitests_JniTests_callAddA(JNIEnv* env,
                                                                    jclass clazz,
                                                                    jint x,
                                                                    jint y) {
  jmethodID method_id = env->GetStaticMethodID(clazz, "add", "(II)I");
  jvalue args[2];
  args[0].i = x;
  args[1].i = y;
  return env->CallStaticIntMethodA(clazz, method_id, args);
}

JNIEXPORT jint JNICALL Java_com_berberis_jnitests_JniTests_callCallIntFromJNI(JNIEnv* env,
                                                                              jclass clazz) {
  jmethodID method_id = env->GetStaticMethodID(clazz, "callIntFromJNI", "()I");
  return env->CallStaticIntMethod(clazz, method_id);
}

// Prevent clang-format form unfolding it into 125+ lines.
// clang-format off
JNIEXPORT jint JNICALL Java_com_berberis_jnitests_JniTests_Sum125(
    JNIEnv*,
    jclass,
    jint arg1, jint arg2, jint arg3, jint arg4, jint arg5, jint arg6, jint arg7, jint arg8,
    jint arg9, jint arg10, jint arg11, jint arg12, jint arg13, jint arg14, jint arg15, jint arg16,
    jint arg17, jint arg18, jint arg19, jint arg20, jint arg21, jint arg22, jint arg23, jint arg24,
    jint arg25, jint arg26, jint arg27, jint arg28, jint arg29, jint arg30, jint arg31, jint arg32,
    jint arg33, jint arg34, jint arg35, jint arg36, jint arg37, jint arg38, jint arg39, jint arg40,
    jint arg41, jint arg42, jint arg43, jint arg44, jint arg45, jint arg46, jint arg47, jint arg48,
    jint arg49, jint arg50, jint arg51, jint arg52, jint arg53, jint arg54, jint arg55, jint arg56,
    jint arg57, jint arg58, jint arg59, jint arg60, jint arg61, jint arg62, jint arg63, jint arg64,
    jint arg65, jint arg66, jint arg67, jint arg68, jint arg69, jint arg70, jint arg71, jint arg72,
    jint arg73, jint arg74, jint arg75, jint arg76, jint arg77, jint arg78, jint arg79, jint arg80,
    jint arg81, jint arg82, jint arg83, jint arg84, jint arg85, jint arg86, jint arg87, jint arg88,
    jint arg89, jint arg90, jint arg91, jint arg92, jint arg93, jint arg94, jint arg95, jint arg96,
    jint arg97, jint arg98, jint arg99, jint arg100, jint arg101, jint arg102, jint arg103,
    jint arg104, jint arg105, jint arg106, jint arg107, jint arg108, jint arg109, jint arg110,
    jint arg111, jint arg112, jint arg113, jint arg114, jint arg115, jint arg116, jint arg117,
    jint arg118, jint arg119, jint arg120, jint arg121, jint arg122, jint arg123, jint arg124,
    jint arg125) {
  // clang-format on
  return arg1 + arg2 + arg3 + arg4 + arg5 + arg6 + arg7 + arg8 + arg9 + arg10 + arg11 + arg12 +
         arg13 + arg14 + arg15 + arg16 + arg17 + arg18 + arg19 + arg20 + arg21 + arg22 + arg23 +
         arg24 + arg25 + arg26 + arg27 + arg28 + arg29 + arg30 + arg31 + arg32 + arg33 + arg34 +
         arg35 + arg36 + arg37 + arg38 + arg39 + arg40 + arg41 + arg42 + arg43 + arg44 + arg45 +
         arg46 + arg47 + arg48 + arg49 + arg50 + arg51 + arg52 + arg53 + arg54 + arg55 + arg56 +
         arg57 + arg58 + arg59 + arg60 + arg61 + arg62 + arg63 + arg64 + arg65 + arg66 + arg67 +
         arg68 + arg69 + arg70 + arg71 + arg72 + arg73 + arg74 + arg75 + arg76 + arg77 + arg78 +
         arg79 + arg80 + arg81 + arg82 + arg83 + arg84 + arg85 + arg86 + arg87 + arg88 + arg89 +
         arg90 + arg91 + arg92 + arg93 + arg94 + arg95 + arg96 + arg97 + arg98 + arg99 + arg100 +
         arg101 + arg102 + arg103 + arg104 + arg105 + arg106 + arg107 + arg108 + arg109 + arg110 +
         arg111 + arg112 + arg113 + arg114 + arg115 + arg116 + arg117 + arg118 + arg119 + arg120 +
         arg121 + arg122 + arg123 + arg124 + arg125;
}

}  // extern "C"
