#!/usr/bin/python
#
# Copyright (C) 2015 The Android Open Source Project
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

import io
import sys
import unittest

import gen_intrinsics


class GenIntrinsicsTests(unittest.TestCase):

  def test_get_semantics_player_hook_proto_smoke(self):
    out = gen_intrinsics._get_semantics_player_hook_proto("Foo", {
        "in": ["uint32_t"],
        "out": []
    })
    self.assertEqual(out, "void Foo(Register arg0)")

  def test_get_semantics_player_hook_proto_template_types(self):
    intr = {
        "Foo": {
            "in": ["uint32_t", "uint8_t", "Type0", "Type1", "vec", "uimm8"],
            "out": ["uint32_t"],
            "class": "template",
            "variants": ["Float32, int32", "Float64, int64"]
        }}
    gen_intrinsics._gen_semantic_player_types(intr.items())
    out = gen_intrinsics._get_semantics_player_hook_proto("Foo", intr["Foo"])
    self.assertEqual(out,
                     "template<typename Type0, typename Type1>\n"
                     "Register Foo(Register arg0, "
                                  "Register arg1, "
                                  "FpRegister arg2, "
                                  "Register arg3, "
                                  "SimdRegister arg4, "
                                  "uint8_t arg5)" ) # pyformat: disable
    out = gen_intrinsics._get_semantics_player_hook_proto("Foo", intr["Foo"], use_type_id=True)
    self.assertEqual(out,
                     "template<intrinsics::TemplateTypeId Type0, intrinsics::TemplateTypeId Type1>\n"
                     "Register Foo(Register arg0, "
                                  "Register arg1, "
                                  "FpRegister arg2, "
                                  "Register arg3, "
                                  "SimdRegister arg4, "
                                  "uint8_t arg5, "
                                  "intrinsics::Value<Type0>, "
                                  "intrinsics::Value<Type1>)" ) # pyformat: disable
    out = gen_intrinsics._get_semantics_player_hook_proto(
        "Foo", intr["Foo"], listener=' Interpreter::')
    self.assertEqual(out,
                     " Interpreter::Register Interpreter::Foo("
                     " Interpreter::Register arg0, "
                     " Interpreter::Register arg1, "
                     " Interpreter::FpRegister arg2, "
                     " Interpreter::Register arg3, "
                     " Interpreter::SimdRegister arg4,"
                     " uint8_t arg5,"
                     " intrinsics::TemplateTypeId Type0,"
                     " intrinsics::TemplateTypeId Type1)" ) # pyformat: disable

  def test_get_semantics_player_hook_proto_operand_types(self):
    out = gen_intrinsics._get_semantics_player_hook_proto(
        "Foo", {
            "in": ["uint32_t", "uint8_t", "Float32", "Float64", "vec", "uimm8"],
            "out": ["uint32_t"]
        })
    self.assertEqual(out,
                     "Register Foo(Register arg0, "
                                  "Register arg1, "
                                  "SimdRegister arg2, "
                                  "SimdRegister arg3, "
                                  "SimdRegister arg4, "
                                  "uint8_t arg5)" ) # pyformat: disable

  def test_get_semantics_player_hook_proto_multiple_results(self):
    out = gen_intrinsics._get_semantics_player_hook_proto("Foo", {
        "in": ["uint32_t"],
        "out": ["vec", "uint32_t"]
    })
    self.assertEqual(out,
                     "std::tuple<SimdRegister, Register> Foo(Register arg0)")

  def test_get_interpreter_hook_call_expr_smoke(self):
    out = gen_intrinsics._get_interpreter_hook_call_expr(
        "Foo", {
            "in": ["uint32_t"],
            "out": []
        })
    self.assertEqual(out, "intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0))")

  def test_get_interpreter_hook_call_expr_template_types(self):
    intr = {
        "Foo": {
            "in": ["uint32_t", "uint8_t", "Type0", "Type1", "vec", "uimm8"],
            "out": ["uint32_t"],
            "class": "template",
            "variants": ["Float32, int32", "Float64, int64"]
        }}
    gen_intrinsics._gen_semantic_player_types(intr.items())
    out = gen_intrinsics._get_interpreter_hook_call_expr("Foo", intr["Foo"])
    self.assertEqual(
        out,
        "IntegerToGPRReg(std::get<0>(intrinsics::Foo<Type0, Type1>("
            "GPRRegToInteger<uint32_t>(arg0), "
            "GPRRegToInteger<uint8_t>(arg1), "
            "FPRegToFloat<Type0>(arg2), "
            "GPRRegToInteger<Type1>(arg3), "
            "arg4, "
            "GPRRegToInteger<uint8_t>(arg5))))" ) # pyforman: disable
    out = gen_intrinsics._get_interpreter_hook_call_expr("Foo", intr["Foo"], use_type_id=True)
    self.assertEqual(
        out,
        "IntegerToGPRReg(std::get<0>(intrinsics::Foo<"
                "intrinsics::TypeFromId<Type0>, intrinsics::TypeFromId<Type1>>("
            "GPRRegToInteger<uint32_t>(arg0), "
            "GPRRegToInteger<uint8_t>(arg1), "
            "FPRegToFloat<intrinsics::TypeFromId<Type0>>(arg2), "
            "GPRRegToInteger<intrinsics::TypeFromId<Type1>>(arg3), "
            "arg4, "
            "GPRRegToInteger<uint8_t>(arg5))))" ) # pyforman: disable

  def test_get_interpreter_hook_call_expr_operand_types(self):
    out = gen_intrinsics._get_interpreter_hook_call_expr(
        "Foo", {
            "in": ["uint32_t", "uint8_t", "Float32", "Float64", "vec", "uimm8"],
            "out": []
        })
    self.assertEqual(out,
                     "intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0), "
                                     "GPRRegToInteger<uint8_t>(arg1), "
                                     "FPRegToFloat<Float32>(arg2), "
                                     "FPRegToFloat<Float64>(arg3), "
                                     "arg4, "
                                     "GPRRegToInteger<uint8_t>(arg5))" ) # pyforman: disable

  def test_get_interpreter_hook_call_expr_single_result(self):
    out = gen_intrinsics._get_interpreter_hook_call_expr(
        "Foo", {
            "in": ["uint32_t"],
            "out": ["uint32_t"]
        })
    self.assertEqual(out, "std::get<0>(intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0)))")

  def test_get_interpreter_hook_call_expr_multiple_result(self):
    out = gen_intrinsics._get_interpreter_hook_call_expr(
        "Foo", {
            "in": ["uint32_t"],
            "out": ["vec", "uint32_t"]
        })
    self.assertEqual(out, "intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0))")

  def test_get_interpreter_hook_call_expr_float32_result(self):
    out = gen_intrinsics._get_interpreter_hook_call_expr(
        "Foo", {
            "in": ["uint32_t"],
            "out": ["Float32"]
        })
    self.assertEqual(out, "FloatToFPReg(std::get<0>(intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0))))")

  def test_get_interpreter_hook_call_expr_float64_result(self):
    out = gen_intrinsics._get_interpreter_hook_call_expr(
        "Foo", {
            "in": ["uint32_t"],
            "out": ["Float64"]
        })
    self.assertEqual(out, "FloatToFPReg(std::get<0>(intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0))))")

  def test_get_interpreter_hook_call_expr_precise_nan(self):
    out = gen_intrinsics._get_interpreter_hook_call_expr(
        "Foo", {
            "in": ["uint32_t"],
            "out": [],
            "precise_nans": True,
        })
    self.assertEqual(
        out, "intrinsics::Foo<config::kPreciseNaNOperationsHandling>("
             "GPRRegToInteger<uint32_t>(arg0))")

  def test_gen_interpreter_hook_return_stmt(self):
    out = gen_intrinsics._get_interpreter_hook_return_stmt(
        "Foo", {
            "in": ["uint32_t"],
            "out": ["uint32_t"]
        })
    self.assertEqual(out, "return std::get<0>(intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0)));")

  def test_gen_interpreter_hook_return_stmt_void(self):
    out = gen_intrinsics._get_interpreter_hook_return_stmt(
        "Foo", {
            "in": ["uint32_t"],
            "out": []
        })
    self.assertEqual(out, "return intrinsics::Foo(GPRRegToInteger<uint32_t>(arg0));")


  def test_get_semantics_player_hook_proto_raw_variant(self):
    out = gen_intrinsics._get_semantics_player_hook_proto(
        "Foo", {
            "class": "vector_8/16",
            "in": ["vec", "vec"],
            "out": ["vec"],
            "variants": ["raw"],
        })
    self.assertEqual(out, "SimdRegister Foo(uint8_t size, "
                                           "SimdRegister arg0, "
                                           "SimdRegister arg1)") # pyformat: disable


  def test_get_interpreter_hook_raw_vector_body(self):
    out = gen_intrinsics._get_semantics_player_hook_raw_vector_body(
        "Foo", {
            "class": "vector_8/16",
            "in": ["vec", "vec"],
            "out": ["vec"],
        }, gen_intrinsics._get_interpreter_hook_return_stmt)
    self.assertSequenceEqual(list(out),
                             ("switch (size) {",
                              "  case 64:" ,
                              "    return std::get<0>(intrinsics::Foo<64>(arg0, arg1));",
                              "  case 128:",
                              "    return std::get<0>(intrinsics::Foo<128>(arg0, arg1));",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported size\");",
                              "}")) # pyformat: disable

  def test_get_interpreter_hook_vector_body_fp(self):
    out = gen_intrinsics._get_interpreter_hook_vector_body(
        "Foo", {
            "class": "vector_16",
            "in": ["vec", "vec"],
            "out": ["vec"],
            "variants": ["Float32"],
        })
    self.assertSequenceEqual(list(out),
                             ("auto format = intrinsics::GetVectorFormatFP(elem_size, elem_num);",
                              "switch (format) {",
                              "  case intrinsics::kVectorF32x4:" ,
                              "    return std::get<0>(intrinsics::Foo<Float32, 4>(arg0, arg1));",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported format\");",
                              "}")) # pyformat: disable


  def test_get_interpreter_hook_vector_body_unsigned_int(self):
    out = gen_intrinsics._get_interpreter_hook_vector_body(
        "Foo", {
            "class": "vector_16",
            "in": ["vec", "vec"],
            "out": ["vec"],
            "variants": ["unsigned_16/32/64"],
        })
    self.assertSequenceEqual(list(out),
                             ("auto format = intrinsics::GetVectorFormatInt(elem_size, elem_num, false);",
                              "switch (format) {",
                              "  case intrinsics::kVectorU16x8:" ,
                              "    return std::get<0>(intrinsics::Foo<uint16_t, 8>(arg0, arg1));",
                              "  case intrinsics::kVectorU32x4:" ,
                              "    return std::get<0>(intrinsics::Foo<uint32_t, 4>(arg0, arg1));",
                              "  case intrinsics::kVectorU64x2:" ,
                              "    return std::get<0>(intrinsics::Foo<uint64_t, 2>(arg0, arg1));",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported format\");",
                              "}")) # pyformat: disable


  def test_get_interpreter_hook_vector_body_signed_int(self):
    out = gen_intrinsics._get_interpreter_hook_vector_body(
        "Foo", {
            "class": "vector_16",
            "in": ["vec", "vec"],
            "out": ["vec"],
            "variants": ["signed_16/32"],
        })
    self.assertSequenceEqual(list(out),
                             ("auto format = intrinsics::GetVectorFormatInt(elem_size, elem_num, true);",
                              "switch (format) {",
                              "  case intrinsics::kVectorI16x8:" ,
                              "    return std::get<0>(intrinsics::Foo<int16_t, 8>(arg0, arg1));",
                              "  case intrinsics::kVectorI32x4:" ,
                              "    return std::get<0>(intrinsics::Foo<int32_t, 4>(arg0, arg1));",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported format\");",
                              "}")) # pyformat: disable


  def test_get_interpreter_hook_vector_body_signed_and_unsigned_int(self):
    out = gen_intrinsics._get_interpreter_hook_vector_body(
        "Foo", {
            "class": "vector_16",
            "in": ["vec", "vec"],
            "out": ["vec"],
            "variants": ["signed_32", "unsigned_32"],
        })
    self.assertSequenceEqual(list(out),
                             ("auto format = intrinsics::GetVectorFormatInt(elem_size, elem_num, is_signed);",
                              "switch (format) {",
                              "  case intrinsics::kVectorI32x4:" ,
                              "    return std::get<0>(intrinsics::Foo<int32_t, 4>(arg0, arg1));",
                              "  case intrinsics::kVectorU32x4:" ,
                              "    return std::get<0>(intrinsics::Foo<uint32_t, 4>(arg0, arg1));",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported format\");",
                              "}")) # pyformat: disable


  def test_get_interpreter_hook_vector_body_vector_8(self):
    out = gen_intrinsics._get_interpreter_hook_vector_body(
        "Foo", {
            "class": "vector_8",
            "in": ["vec", "vec"],
            "out": ["vec"],
            "variants": ["unsigned_32"],
        })
    self.assertSequenceEqual(list(out),
                             ("auto format = intrinsics::GetVectorFormatInt(elem_size, elem_num, false);",
                              "switch (format) {",
                              "  case intrinsics::kVectorU32x2:" ,
                              "    return std::get<0>(intrinsics::Foo<uint32_t, 2>(arg0, arg1));",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported format\");",
                              "}")) # pyformat: disable


  def test_get_interpreter_hook_vector_body_single(self):
    out = gen_intrinsics._get_interpreter_hook_vector_body(
        "Foo", {
            "class": "vector_8/16/single",
            "variants": ["signed_32"],
            "in": ["vec", "fp_flags"],
            "out": ["vec", "fp_flags"],
        })
    self.assertSequenceEqual(list(out),
                             ("auto format = intrinsics::GetVectorFormatInt(elem_size, elem_num, true);",
                              "switch (format) {",
                              "  case intrinsics::kVectorI32x2:" ,
                              "    return intrinsics::Foo<int32_t, 2>(arg0, GPRRegToInteger<uint32_t>(arg1));",
                              "  case intrinsics::kVectorI32x4:" ,
                              "    return intrinsics::Foo<int32_t, 4>(arg0, GPRRegToInteger<uint32_t>(arg1));",
                              "  case intrinsics::kVectorI32x1:" ,
                              "    return intrinsics::Foo<int32_t, 1>(arg0, GPRRegToInteger<uint32_t>(arg1));",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported format\");",
                              "}")) # pyformat: disable


  def test_get_translator_hook_call_expr_smoke(self):
    out = gen_intrinsics._get_translator_hook_call_expr(
        "Foo", {
            "in": ["uint32_t"],
            "out": ["uint32_t"],
        })
    self.assertEqual(out, "CallIntrinsic<&intrinsics::Foo, Register>(arg0)")


  def test_get_translator_hook_call_expr_void(self):
    out = gen_intrinsics._get_translator_hook_call_expr(
        "Foo", {
            "in": [],
            "out": [],
        })
    self.assertEqual(out, "CallIntrinsic<&intrinsics::Foo, void>()")


  def test_get_translator_hook_raw_vector_body(self):
    out = gen_intrinsics._get_semantics_player_hook_raw_vector_body(
        "Foo", {
            "class": "vector_8/16",
            "in": ["vec", "vec"],
            "out": ["vec"],
        }, gen_intrinsics._get_translator_hook_return_stmt)
    self.assertSequenceEqual(list(out),
                             ("switch (size) {",
                              "  case 64:",
                              "    return CallIntrinsic<&intrinsics::Foo<64>, SimdRegister>(arg0, arg1);",
                              "  case 128:",
                              "    return CallIntrinsic<&intrinsics::Foo<128>, SimdRegister>(arg0, arg1);",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported size\");",
                              "}")) # pyformat: disable


  def test_get_translator_hook_vector_body(self):
    out = gen_intrinsics._get_semantics_player_hook_vector_body(
        "Foo", {
            "class": "vector_8/16/single",
            "variants": ["signed_32"],
            "in": ["vec", "fp_flags"],
            "out": ["vec", "fp_flags"],
        }, gen_intrinsics._get_translator_hook_return_stmt)
    self.assertSequenceEqual(list(out),
                             ("auto format = intrinsics::GetVectorFormatInt(elem_size, elem_num, true);",
                              "switch (format) {",
                              "  case intrinsics::kVectorI32x2:" ,
                              "    return CallIntrinsic<&intrinsics::Foo<int32_t, 2>, std::tuple<SimdRegister, Register>>(arg0, arg1);",
                              "  case intrinsics::kVectorI32x4:" ,
                              "    return CallIntrinsic<&intrinsics::Foo<int32_t, 4>, std::tuple<SimdRegister, Register>>(arg0, arg1);",
                              "  case intrinsics::kVectorI32x1:" ,
                              "    return CallIntrinsic<&intrinsics::Foo<int32_t, 1>, std::tuple<SimdRegister, Register>>(arg0, arg1);",
                              "  default:",
                              "    LOG_ALWAYS_FATAL(\"Unsupported format\");",
                              "}")) # pyformat: disable


  def test_gen_template_parameters_verifier(self):
    intrinsic = {
            "class": "template",
            "variants": [ "int32_t", "int64_t" ],
            "in": [ "Type0", "int8_t" ],
            "out": [ "Type0" ]
        }
    out = io.StringIO()
    gen_intrinsics._gen_template_parameters_verifier(out, intrinsic)
    self.assertSequenceEqual(out.getvalue(),
                             "  static_assert(std::tuple{intrinsics::kIdFromType<Type0>} == "
                             "std::tuple{intrinsics::kIdFromType<int32_t>} || "
                             "std::tuple{intrinsics::kIdFromType<Type0>} == "
                             "std::tuple{intrinsics::kIdFromType<int64_t>});\n") # pyformat: disable
    out = io.StringIO()
    gen_intrinsics._gen_template_parameters_verifier(out, intrinsic, use_type_id=True)
    self.assertSequenceEqual(out.getvalue(),
                             "  static_assert(std::tuple{Type0} == "
                             "std::tuple{intrinsics::kIdFromType<int32_t>} || std::tuple{Type0} == "
                             "std::tuple{intrinsics::kIdFromType<int64_t>});\n") # pyformat: disable

  def test_gen_interpreter_hook(self):
    intrinsic = {
            "class": "template",
            "variants": [ "int32_t", "int64_t" ],
            "in": [ "Type0", "int8_t" ],
            "out": [ "Type0" ]
        }
    out = io.StringIO()
    gen_intrinsics._gen_interpreter_hook(out, "Foo", intrinsic, '')
    self.assertSequenceEqual(out.getvalue(),
                             "template<typename Type0>\n"
                             "Register Foo(Register arg0, Register arg1) const {\n"
                             "  return std::get<0>(intrinsics::Foo<Type0>(GPRRegToInteger<Type0>(arg0), "
                             "GPRRegToInteger<int8_t>(arg1)));\n"
                             "}\n\n") # pyformat: disable
    out = io.StringIO()
    gen_intrinsics._gen_interpreter_hook(out, "Foo", intrinsic, '', use_type_id=True)
    self.assertSequenceEqual(out.getvalue(),
                             "#ifndef BERBERIS_INTRINSICS_HOOKS_INLINE_DEMULTIPLEXER\n"
                             " Register Foo( Register arg0,  Register arg1, intrinsics::TemplateTypeId "
                             "Type0) const;\n"
                             "#endif\n"
                             "template<intrinsics::TemplateTypeId Type0>\n"
                             "Register Foo(Register arg0, Register arg1, intrinsics::Value<Type0>) const {\n"
                             "  return std::get<0>(intrinsics::Foo<intrinsics::TypeFromId<Type0>>("
                             "GPRRegToInteger<intrinsics::TypeFromId<Type0>>(arg0), "
                             "GPRRegToInteger<int8_t>(arg1)));\n"
                             "}\n\n") # pyformat: disable

  def test_gen_demultiplexer_hook(self):
    intrinsic = {
            "class": "template",
            "variants": [ "int32_t", "int64_t" ],
            "in": [ "Type0", "int8_t" ],
            "out": [ "Type0" ]
        }
    out = io.StringIO()
    gen_intrinsics._gen_demultiplexer_hook(out, "Foo", intrinsic)
    self.assertSequenceEqual(out.getvalue(),
                             " BERBERIS_INTRINSICS_HOOKS_LISTENER Register BERBERIS_INTRINSICS_HOOKS_LISTENER "
                             "Foo( BERBERIS_INTRINSICS_HOOKS_LISTENER Register arg0,  "
                             "BERBERIS_INTRINSICS_HOOKS_LISTENER Register arg1, intrinsics::TemplateTypeId "
                             "Type0) BERBERIS_INTRINSICS_HOOKS_CONST {\n"
                             "  switch (intrinsics::TrivialDemultiplexer(Type0)) {\n"
                             "    case "
                             "intrinsics::TrivialDemultiplexer(intrinsics::kIdFromType<int32_t>):\n"
                             "      // Disable LOG_NDEBUG to use DCHECK for debugging!\n"
                             "      DCHECK_EQ(intrinsics::kIdFromType<int32_t>, Type0);\n"
                             "      return Foo<int32_t>(arg0,arg1);\n"
                             "    case "
                             "intrinsics::TrivialDemultiplexer(intrinsics::kIdFromType<int64_t>):\n"
                             "      // Disable LOG_NDEBUG to use DCHECK for debugging!\n"
                             "      DCHECK_EQ(intrinsics::kIdFromType<int64_t>, Type0);\n"
                             "      return Foo<int64_t>(arg0,arg1);\n"
                             "    default:\n"
                             "      FATAL(\"Unsupported size\");\n"
                             "  }\n"
                             "}\n\n") # pyformat: disable


  def test_gen_demultiplexer_hook_multiple_types(self):
    intrinsic = {
            "class": "template",
            "variants": [ "int32_t, Float32", "int64_t, Float64" ],
            "in": [ "Type0", "int8_t" ],
            "out": [ "Type0" ]
        }
    out = io.StringIO()
    gen_intrinsics._gen_demultiplexer_hook(out, "Foo", intrinsic)
    self.assertSequenceEqual(out.getvalue(),
                             " BERBERIS_INTRINSICS_HOOKS_LISTENER Register "
                             "BERBERIS_INTRINSICS_HOOKS_LISTENER Foo( BERBERIS_INTRINSICS_HOOKS_LISTENER "
                             "Register arg0,  BERBERIS_INTRINSICS_HOOKS_LISTENER Register arg1, "
                             "intrinsics::TemplateTypeId Type0, intrinsics::TemplateTypeId Type1) "
                             "BERBERIS_INTRINSICS_HOOKS_CONST {\n"
                             "  switch (intrinsics::TrivialDemultiplexer(Type0, Type1)) {\n"
                             "    case intrinsics::TrivialDemultiplexer(intrinsics::kIdFromType<int32_t>, "
                             "intrinsics::kIdFromType<Float32>):\n"
                             "      // Disable LOG_NDEBUG to use DCHECK for debugging!\n"
                             "      DCHECK_EQ(intrinsics::kIdFromType<int32_t>, Type0);\n"
                             "      DCHECK_EQ(intrinsics::kIdFromType<Float32>, Type1);\n"
                             "      return Foo<int32_t, Float32>(arg0,arg1);\n"
                             "    case intrinsics::TrivialDemultiplexer(intrinsics::kIdFromType<int64_t>, "
                             "intrinsics::kIdFromType<Float64>):\n"
                             "      // Disable LOG_NDEBUG to use DCHECK for debugging!\n"
                             "      DCHECK_EQ(intrinsics::kIdFromType<int64_t>, Type0);\n"
                             "      DCHECK_EQ(intrinsics::kIdFromType<Float64>, Type1);\n"
                             "      return Foo<int64_t, Float64>(arg0,arg1);\n"
                             "    default:\n"
                             "      FATAL(\"Unsupported size\");\n"
                             "  }\n"
                             "}\n\n") # pyformat: disable


if __name__ == "__main__":
  unittest.main(verbosity=2)
