// Copyright (C) 2024 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

.section .text
.globl _start
_start:
    // SYS_write
    li a7, 64
    // fd is stdout
    li a0, 1
    // pc-relative address of the string
1:
    auipc a1, %pcrel_hi(hello_string)
    addi a1, a1, %pcrel_lo(1b)
    // string size
    li a2, 6
    ecall

    // SYS_exit
    li a7, 93
    // value to return from main
    li a0, 0
    ecall

.section .rodata
hello_string:
   .asciz "Hello\n"
