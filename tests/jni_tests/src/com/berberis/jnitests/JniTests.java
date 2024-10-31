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

package com.berberis.jnitests;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public final class JniTests {
    static {
        System.loadLibrary("berberis_jni_tests");
    }

    static native int intFromJNI();

    @Test
    public void testReturnInt() {
      assertEquals(42, intFromJNI());
    }

    static native boolean isJNIOnLoadCalled();

    @Test
    public void testOnLoadCalled() {
      assertTrue(isJNIOnLoadCalled());
    }

    static native boolean checkGetVersion();

    @Test
    public void testGetVersion() {
      assertTrue(checkGetVersion());
    }

    static native boolean checkJavaVMCorrespondsToJNIEnv();

    @Test
    public void testJavaVMCorrespondsToJNIEnv() {
      assertTrue(checkJavaVMCorrespondsToJNIEnv());
    }

    static native boolean callRegisterNatives();
    static native int add42(int x);

    @Test
    public void testRegisterNatives() {
      assertTrue(callRegisterNatives());
      assertEquals(84, add42(42));
    }

    static int add(int x, int y) {
      return x + y;
    }

    static native int callAdd(int x, int y);

    @Test
    public void testCallStaticIntMethod() {
      assertEquals(84, callAdd(42, 42));
    }

    static native int callAddA(int x, int y);

    @Test
    public void testCallStaticIntMethodA() {
      assertEquals(84, callAddA(42, 42));
    }

    static int callIntFromJNI() {
      return intFromJNI();
    }

    static native int callCallIntFromJNI();

    @Test
    public void testCallNativeCallJavaCallNative() {
      assertEquals(42, callCallIntFromJNI());
    }

    @Test
    public void testCallNativeMethodWith125Args() {
        assertEquals(5250, Sum125(
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42));
    }

    static native int Sum125(
            int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8,
            int arg9, int arg10, int arg11, int arg12, int arg13, int arg14, int arg15, int arg16,
            int arg17, int arg18, int arg19, int arg20, int arg21, int arg22, int arg23, int arg24,
            int arg25, int arg26, int arg27, int arg28, int arg29, int arg30, int arg31, int arg32,
            int arg33, int arg34, int arg35, int arg36, int arg37, int arg38, int arg39, int arg40,
            int arg41, int arg42, int arg43, int arg44, int arg45, int arg46, int arg47, int arg48,
            int arg49, int arg50, int arg51, int arg52, int arg53, int arg54, int arg55, int arg56,
            int arg57, int arg58, int arg59, int arg60, int arg61, int arg62, int arg63, int arg64,
            int arg65, int arg66, int arg67, int arg68, int arg69, int arg70, int arg71, int arg72,
            int arg73, int arg74, int arg75, int arg76, int arg77, int arg78, int arg79, int arg80,
            int arg81, int arg82, int arg83, int arg84, int arg85, int arg86, int arg87, int arg88,
            int arg89, int arg90, int arg91, int arg92, int arg93, int arg94, int arg95, int arg96,
            int arg97, int arg98, int arg99, int arg100, int arg101, int arg102, int arg103,
            int arg104, int arg105, int arg106, int arg107, int arg108, int arg109, int arg110,
            int arg111, int arg112, int arg113, int arg114, int arg115, int arg116, int arg117,
            int arg118, int arg119, int arg120, int arg121, int arg122, int arg123, int arg124,
            int arg125);
}
