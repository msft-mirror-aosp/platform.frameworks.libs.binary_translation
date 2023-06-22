#!/usr/bin/python3
#
# Copyright (C) 2018 The Android Open Source Project
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

"""Generate intrinsics code."""

from collections import OrderedDict

import asm_defs
import json
import os
import sys

# C-level intrinsic calling convention:
# 1. All arguments are passed using the natural data types:
#  - int8_t passed as one byte argument (on the stack in IA32 mode, in GP register in x86-64 mode)
#  - int32_t passed as 4 bytes argument (on the stack in IA32 mode, in GP register in x86-64 mode)
#  - int64_t is passed as 8 byte argument (on the stack in IA32 mode, in GP register in x86-64 mode)
#  - float is passed as float (on the stack in IA32 mode, in XMM register in x86-64 mode)
#  - double is passed as double (on the stack in IA32 mode, in XMM register in x86-64 mode)
#  - vector formats are passed as pointers to 128bit data structure
# 2. Return values.
#  - Values are returned as std::tuple.  This means that on IA32 it's always returned on stack.

INDENT = '  '
AUTOGEN = """\
// This file automatically generated by gen_intrinsics.py
// DO NOT EDIT!
"""


class VecFormat(object):

  def __init__(self, num_elements, element_size, is_unsigned, is_float, index,
               c_type):
    self.num_elements = num_elements
    self.element_size = element_size
    self.is_unsigned = is_unsigned
    self.is_float = is_float
    self.index = index
    self.c_type = c_type


# Vector format defined as:
#  vector_size, element_size, is_unsigned, is_float, index, ir_format, c_type
# TODO(olonho): make flat numbering after removing legacy macro compat.
_VECTOR_FORMATS = {
    'U8x8': VecFormat(8, 1, True, False, 1, 'uint8_t'),
    'U16x4': VecFormat(4, 2, True, False, 2, 'uint16_t'),
    'U32x2': VecFormat(2, 4, True, False, 3, 'uint32_t'),
    'U64x1': VecFormat(1, 8, True, False, 4, 'uint64_t'),
    'U8x16': VecFormat(16, 1, True, False, 5, 'uint8_t'),
    'U16x8': VecFormat(8, 2, True, False, 6, 'uint16_t'),
    'U32x4': VecFormat(4, 4, True, False, 7, 'uint32_t'),
    'U64x2': VecFormat(2, 8, True, False, 8, 'uint64_t'),
    'I8x8': VecFormat(8, 1, False, False, 9, 'int8_t'),
    'I16x4': VecFormat(4, 2, False, False, 10, 'int16_t'),
    'I32x2': VecFormat(2, 4, False, False, 11, 'int32_t'),
    'I64x1': VecFormat(1, 8, False, False, 12, 'int64_t'),
    'I8x16': VecFormat(16, 1, False, False, 13, 'int8_t'),
    'I16x8': VecFormat(8, 2, False, False, 14, 'int16_t'),
    'I32x4': VecFormat(4, 4, False, False, 15, 'int32_t'),
    'I64x2': VecFormat(2, 8, False, False, 16, 'int64_t'),
    'U8x1': VecFormat(1, 1, True, False, 17, 'uint8_t'),
    'I8x1': VecFormat(1, 1, False, False, 18, 'int8_t'),
    'U16x1': VecFormat(1, 2, True, False, 19, 'uint16_t'),
    'I16x1': VecFormat(1, 2, False, False, 20, 'int16_t'),
    'U32x1': VecFormat(1, 4, True, False, 21, 'uint32_t'),
    'I32x1': VecFormat(1, 4, False, False, 22, 'int32_t'),
    # These vector formats can never intersect with above, so can reuse index.
    'F32x1': VecFormat(1, 4, False, True, 1, 'Float32'),
    'F32x2': VecFormat(2, 4, False, True, 2, 'Float32'),
    'F32x4': VecFormat(4, 4, False, True, 3, 'Float32'),
    'F64x1': VecFormat(1, 8, False, True, 4, 'Float64'),
    'F64x2': VecFormat(2, 8, False, True, 5, 'Float64'),
    # Those vector formats can never intersect with above, so can reuse index.
    'U8x4': VecFormat(4, 1, True, False, 1, 'uint8_t'),
    'U16x2': VecFormat(2, 2, True, False, 2, 'uint16_t'),
    'I8x4': VecFormat(4, 1, False, False, 3, 'int8_t'),
    'I16x2': VecFormat(2, 2, False, False, 4, 'int16_t'),
}


class VecSize(object):

  def __init__(self, num_elements, index):
    self.num_elements = num_elements
    self.index = index


_VECTOR_SIZES = {'X8': VecSize(8, 1), 'X16': VecSize(16, 2)}


def _is_imm_type(arg_type):
  return 'imm' in arg_type


def _get_imm_c_type(arg_type):
  return {
      'imm8' : 'int8_t',
      'uimm8' : 'uint8_t',
      'uimm32' : 'uint32_t',
  }[arg_type]


def _get_c_type(arg_type):
  if arg_type in ('Float32', 'Float64', 'int8_t', 'uint8_t', 'int16_t',
                  'uint16_t', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t',
                  'volatile uint8_t*', 'volatile uint32_t*'):
    return arg_type
  if arg_type in ('fp_flags', 'fp_control', 'int', 'flag', 'flags', 'vec32'):
    return 'uint32_t'
  if _is_imm_type(arg_type):
    return _get_imm_c_type(arg_type)
  if arg_type == 'vec':
    return 'SIMD128Register'
  raise Exception('Type %s not supported' % (arg_type))


def _get_semantic_player_type(arg_type):
  if arg_type in ('Float32', 'Float64', 'vec'):
    return 'SimdRegister'
  if _is_imm_type(arg_type):
    return _get_imm_c_type(arg_type)
  return 'Register'


def _gen_scalar_intr_decl(f, name, intr):
  ins = intr.get('in')
  outs = intr.get('out')
  params = [_get_c_type(op) for op in ins]
  if len(outs) > 0:
    retval = 'std::tuple<' + ', '.join(_get_c_type(out) for out in outs) + '>'
  else:
    retval = 'void'
  comment = intr.get('comment')
  if comment:
    print('// %s.' % (comment), file=f)
  if intr.get('precise_nans', False):
    print('template <bool precise_nan_operations_handling, '
          'enum PreferCppImplementation = kUseAssemblerImplementationIfPossible>',
          file=f)
  print('%s %s(%s);' % (retval, name, ', '.join(params)), file=f)


def _is_vector_class(intr):
  return intr.get('class') in ('vector_4', 'vector_8', 'vector_16',
                               'vector_8/16', 'vector_8/16/single',
                               'vector_8/single', 'vector_16/single')


def _is_simd128_conversion_required(t):
  return (_get_semantic_player_type(t) == 'SimdRegister' and
          _get_c_type(t) != 'SIMD128Register')


def _get_semantics_player_hook_result(intr):
  outs = intr['out']
  if len(outs) == 0:
    return 'void'
  elif len(outs) == 1:
    # No tuple for single result.
    return _get_semantic_player_type(outs[0])
  return 'std::tuple<' + ', '.join(
      _get_semantic_player_type(out) for out in outs) + '>'


def _get_semantics_player_hook_proto_components(name, intr):
  ins = intr['in']

  args = []
  if _is_vector_class(intr):
    if 'raw' in intr['variants']:
      assert len(intr['variants']) == 1, "Unexpected length of variants"
      args = ["uint8_t size"]
    else:
      args = ["uint8_t elem_size", "uint8_t elem_num"]
      if (_is_signed(intr) and _is_unsigned(intr)):
        args += ['bool is_signed']

  args += [
      '%s arg%d' % (_get_semantic_player_type(op), num)
      for num, op in enumerate(ins)
  ]

  result = _get_semantics_player_hook_result(intr)

  return result, name, ', '.join(args)


def _get_semantics_player_hook_proto(name, intr):
  result, name, args = _get_semantics_player_hook_proto_components(name, intr)
  return '%s %s(%s)' % (result, name, args)


def _get_interpreter_hook_call_expr(name, intr, desc=None):
  ins = intr['in']
  outs = intr['out']

  call_params = []
  for num, op in enumerate(ins):
    arg = 'arg%d' % (num)
    if _get_semantic_player_type(op) == 'SimdRegister':
      call_params.append(_get_cast_from_simd128(arg, op, ptr_bits=64))
    elif '*' in _get_c_type(op):
      call_params.append('bit_cast<%s>(%s)' % (_get_c_type(op), arg))
    else:
      call_params.append(arg)

  call_expr = 'intrinsics::%s%s(%s)' % (
      name, _get_desc_specializations(intr, desc).replace(
          'Float', 'intrinsics::Float'), ', '.join(call_params))

  if (len(outs) == 1):
    # Unwrap tuple for single result.
    call_expr = 'std::get<0>(%s)' % call_expr
    # Currently this kind of mismatch can only happen for single result, so we
    # can keep simple code here for now.
    if (_is_simd128_conversion_required(outs[0])):
      out_type = _get_c_type(outs[0])
      if out_type in ('Float32', 'Float64'):
        call_expr = 'SimdRegister(%s)' % call_expr
      else:
        raise Exception('Type %s is not supported' % (out_type))
  else:
    if (any(_is_simd128_conversion_required(out) for out in outs)):
      raise Exception(
          'Unsupported SIMD128Register conversion with multiple results')

  return call_expr


def _get_interpreter_hook_return_stmt(name, intr, desc=None):
  return 'return ' + _get_interpreter_hook_call_expr(name, intr, desc) + ';'


def _get_semantics_player_hook_raw_vector_body(name, intr, get_return_stmt):
  outs = intr['out']
  if (len(outs) == 0):
    raise Exception('No result raw vector intrinsic is not supported')
  reg_class = intr.get('class')
  yield 'switch (size) {'
  for fmt, desc in _VECTOR_SIZES.items():
    if _check_reg_class_size(reg_class, desc.num_elements):
      yield INDENT + 'case %s:' % desc.num_elements
      yield 2 * INDENT + get_return_stmt(name, intr, desc)
  yield INDENT + 'default:'
  yield 2 * INDENT + 'LOG_ALWAYS_FATAL("Unsupported size");'
  yield 2 * INDENT + 'return {};'
  yield '}'


def _is_signed(intr):
  return any(v.startswith("signed") for v in intr['variants'])


def _is_unsigned(intr):
  return any(v.startswith("unsigned") for v in intr['variants'])


def _get_vector_format_init_expr(intr):
  variants = intr.get('variants')

  if ('Float32' in variants or 'Float64' in variants):
    return 'intrinsics::GetVectorFormatFP(elem_size, elem_num)'

  assert _is_signed(intr) or _is_unsigned(intr), "Unexpected intrinsic class"
  if _is_signed(intr) and _is_unsigned(intr):
    signed_arg = ', is_signed'
  else:
    signed_arg = ', true' if _is_signed(intr) else ', false'
  return 'intrinsics::GetVectorFormatInt(elem_size, elem_num%s)' % signed_arg


def _get_semantics_player_hook_vector_body(name, intr, get_return_stmt):
  outs = intr['out']
  if (len(outs) == 0):
    raise Exception('No result vector intrinsic is not supported')
  reg_class = intr.get('class')
  yield 'auto format = %s;' % _get_vector_format_init_expr(intr)
  yield 'switch (format) {'
  for variant in intr.get('variants'):
    for fmt, desc in _VECTOR_FORMATS.items():
      if (_check_reg_class_size(reg_class,
                                desc.element_size * desc.num_elements) and
          _check_typed_variant(variant, desc)):
        yield INDENT + 'case intrinsics::kVector%s:' % fmt
        yield 2 * INDENT + get_return_stmt(name, intr, desc)
      elif (reg_class in ('vector_8/single', 'vector_8/16/single', 'vector_16/single') and
            desc.num_elements == 1 and
          _check_typed_variant(variant, desc)):
        assert desc.element_size <= 8, "Unexpected element size"
        yield INDENT + 'case intrinsics::kVector%s:' % fmt
        yield 2 * INDENT + get_return_stmt(name, intr, desc)
  yield INDENT + 'default:'
  yield 2 * INDENT + 'LOG_ALWAYS_FATAL("Unsupported format");'
  yield 2 * INDENT + 'return {};'
  yield '}'


# Syntax sugar heavily used in tests.
def _get_interpreter_hook_vector_body(name, intr):
  return _get_semantics_player_hook_vector_body(
      name, intr, _get_interpreter_hook_return_stmt)


def _gen_interpreter_hook(f, name, intr):
  print('%s {' % (_get_semantics_player_hook_proto(name, intr)), file=f)

  if _is_vector_class(intr):
    if 'raw' in intr['variants']:
      assert len(intr['variants']) == 1, "Unexpected length of variants"
      lines = _get_semantics_player_hook_raw_vector_body(
          name,
          intr,
          _get_interpreter_hook_return_stmt)
    else:
      lines = _get_interpreter_hook_vector_body(name, intr)

    lines = [INDENT + l for l in lines]
    print('\n'.join(lines), file=f)
  else:
    print(INDENT + _get_interpreter_hook_return_stmt(name, intr), file=f)

  print('}\n', file=f)


def _get_translator_hook_call_expr(name, intr, desc = None):
  desc_spec = _get_desc_specializations(intr, desc).replace(
      'Float', 'intrinsics::Float')
  args = [('arg%d' % n) for n, _ in enumerate(intr['in'])]
  template_params = ['&intrinsics::' + name + desc_spec]
  template_params += [_get_semantics_player_hook_result(intr)]
  return 'CallIntrinsic<%s>(%s)' % (', '.join(template_params), ', '.join(args))


def _get_translator_hook_return_stmt(name, intr, desc=None):
  return 'return ' + _get_translator_hook_call_expr(name, intr, desc) + ';'


def _gen_translator_hook(f, name, intr):
  print('%s {' % (_get_semantics_player_hook_proto(name, intr)), file=f)

  if _is_vector_class(intr):
    if 'raw' in intr['variants']:
      assert len(intr['variants']) == 1, "Unexpected length of variants"
      lines = _get_semantics_player_hook_raw_vector_body(
          name,
          intr,
          _get_translator_hook_return_stmt)
    else:
      lines = _get_semantics_player_hook_vector_body(
          name,
          intr,
          _get_translator_hook_return_stmt)
    lines = [INDENT + l for l in lines]
    print('\n'.join(lines), file=f)
  else:
    print(INDENT + _get_translator_hook_return_stmt(name, intr), file=f)

  print('}\n', file=f)


def _gen_mock_semantics_listener_hook(f, name, intr):
  result, name, args = _get_semantics_player_hook_proto_components(name, intr)
  print('MOCK_METHOD((%s), %s, (%s));' % (result, name, args), file=f)


def _check_signed_variant(variant, desc):
  if variant == 'signed':
    return True
  if variant == 'signed_32':
    return desc.element_size == 4
  if variant == 'signed_64':
    return desc.element_size == 8
  if variant == 'signed_16/32':
    return desc.element_size in (2, 4)
  if variant == 'signed_8/16/32':
    return desc.element_size in (1, 2, 4)
  if variant == 'signed_16/32/64':
    return desc.element_size in (2, 4, 8)
  if variant == 'signed_8/16/32/64':
    return desc.element_size in (1, 2, 4, 8)
  if variant == 'signed_32/64':
    return desc.element_size in (4, 8)
  return False


def _check_unsigned_variant(variant, desc):
  if variant == 'unsigned':
    return True
  if variant == 'unsigned_8':
    return desc.element_size == 1
  if variant == 'unsigned_16':
    return desc.element_size == 2
  if variant == 'unsigned_32':
    return desc.element_size == 4
  if variant == 'unsigned_64':
    return desc.element_size == 8
  if variant == 'unsigned_8/16':
    return desc.element_size in (1, 2)
  if variant == 'unsigned_8/16/32':
    return desc.element_size in (1, 2, 4)
  if variant == 'unsigned_16/32/64':
    return desc.element_size in (2, 4, 8)
  if variant == 'unsigned_8/16/32/64':
    return desc.element_size in (1, 2, 4, 8)
  if variant == 'unsigned_32/64':
    return desc.element_size in (4, 8)
  return False


def _check_reg_class_size(reg_class, size):
  # Small vectors are separate namespace.
  if size == 4 and reg_class == 'vector_4':
    return True
  if size == 8 and reg_class in ('vector_8', 'vector_8/16', 'vector_8/16/single',
                                 'vector_8/single'):
    return True
  if size == 16 and reg_class in ('vector_16', 'vector_8/16', 'vector_8/16/single',
                                  'vector_16/single'):
    return True
  return False


def _check_typed_variant(variant, desc):
  if desc.is_unsigned and not desc.is_float:
    return _check_unsigned_variant(variant, desc)
  if not desc.is_unsigned and not desc.is_float:
    return _check_signed_variant(variant, desc)
  if desc.is_float:
    if desc.element_size == 4:
      return variant == 'Float32'
    if desc.element_size == 8:
      return variant == 'Float64'
  return False


def _get_formats_with_descriptions(intr):
  reg_class = intr.get('class')
  for variant in intr.get('variants'):
    found_fmt = False
    for fmt, desc in _VECTOR_FORMATS.items():
      if (_check_reg_class_size(reg_class,
                                desc.element_size * desc.num_elements) and
          _check_typed_variant(variant, desc) and
          (reg_class != 'vector_4' or desc.element_size < 4)):
        found_fmt = True
        yield fmt, desc

    if variant == 'raw':
      for fmt, desc in _VECTOR_SIZES.items():
        if _check_reg_class_size(reg_class, desc.num_elements):
          found_fmt = True
          yield fmt, desc

    assert found_fmt, 'Couldn\'t expand %s' % reg_class


def _get_result_type(outs):
  result_type = 'void'
  return_stmt = ''
  if len(outs) >= 1:
    result_type = ('std::tuple<' +
                   ', '.join(_get_c_type(out) for out in outs) + '>')
    return_stmt = 'return '
  return result_type, return_stmt


def _get_in_params(params):
  for param_index, param in enumerate(params):
    yield _get_c_type(param), 'in%d' % (param_index)


def _get_out_params(params):
  for param_index, param in enumerate(params):
    yield _get_c_type(param), 'out%d' % (param_index)


def _get_cast_from_simd128(var, target_type, ptr_bits):
  if ('*' in target_type):
    return 'bit_cast<%s>(%s.Get<uint%d_t>(0))' % (_get_c_type(target_type), var,
                                                  ptr_bits)

  cast_map = {
      'Float32': '.Get<intrinsics::Float32>(0)',
      'Float64': '.Get<intrinsics::Float64>(0)',
      'int8_t': '.Get<int8_t>(0)',
      'uint8_t': '.Get<uint8_t>(0)',
      'int16_t': '.Get<int16_t>(0)',
      'uint16_t': '.Get<uint16_t>(0)',
      'int32_t': '.Get<int32_t>(0)',
      'uint32_t': '.Get<uint32_t>(0)',
      'int64_t': '.Get<int64_t>(0)',
      'uint64_t': '.Get<uint64_t>(0)',
      'SIMD128Register': ''
  }
  return '%s%s' % (var, cast_map[_get_c_type(target_type)])


def _get_desc_specializations(intr, desc=None):
  if hasattr(desc, 'c_type'):
    spec = [desc.c_type, str(desc.num_elements)]
  elif hasattr(desc, 'num_elements'):
    spec = [str(desc.num_elements)]
  else:
    spec = []
  if intr.get('precise_nans', False):
    spec = ['Config::kPreciseNaNOperationsHandling'] + spec
  if not len(spec):
    return ''
  return '<%s>' % ', '.join(spec)


def _intr_has_side_effects(intr, fmt=None):
  # If we have 'has_side_effects' mark in JSON file then we use it "as is".
  if 'has_side_effects' in intr:
    return intr.get('has_side_effects')
  # Otherwise we mark all floating-point related intrinsics as "volatile".
  # TODO(b/68857496): move that information in HIR/LIR and stop doing that.
  if 'Float32' in intr.get('in') or 'Float64' in intr.get('in'):
    return True
  if 'Float32' in intr.get('out') or 'Float64' in intr.get('out'):
    return True
  if fmt is not None and fmt.startswith('F'):
    return True
  return False


def _gen_intrinsics_inl_h(f, intrs):
  print(AUTOGEN, file=f)
  for name, intr in intrs:
    if intr.get('class') == 'scalar':
      _gen_scalar_intr_decl(f, name, intr)


def _gen_interpreter_intrinsics_hooks_impl_inl_h(f, intrs):
  print(AUTOGEN, file=f)
  for name, intr in intrs:
    _gen_interpreter_hook(f, name, intr)


def _gen_translator_intrinsics_hooks_impl_inl_h(f, intrs):
  print(AUTOGEN, file=f)
  for name, intr in intrs:
    _gen_translator_hook(f, name, intr)


def _gen_mock_semantics_listener_intrinsics_hooks_impl_inl_h(f, intrs):
  print(AUTOGEN, file=f)
  for name, intr in intrs:
    _gen_mock_semantics_listener_hook(f, name, intr)


def _get_reg_operand_info(arg, info_prefix=None):
  need_tmp = arg['class'] in ('EAX', 'EDX', 'CL', 'ECX')
  if info_prefix is None:
    class_info = 'void'
  else:
    class_info = '%s::%s' % (info_prefix, arg['class'])
  if arg['class'] == 'Imm8':
    return 'ImmArg<%d, int8_t, %s>' % (arg['ir_arg'], class_info)
  if info_prefix is None:
    using_info = 'void'
  else:
    using_info = '%s::%s' % (info_prefix, {
        'def': 'Def',
        'def_early_clobber': 'DefEarlyClobber',
        'use': 'Use',
        'use_def': 'UseDef'
    }[arg['usage']])
  if arg['usage'] == 'use':
    if need_tmp:
      return 'InTmpArg<%d, %s, %s>' % (arg['ir_arg'], class_info, using_info)
    return 'InArg<%d, %s, %s>' % (arg['ir_arg'], class_info, using_info)
  if arg['usage'] in ('def', 'def_early_clobber'):
    assert 'ir_arg' not in arg
    if 'ir_res' in arg:
      if need_tmp:
        return 'OutTmpArg<%d, %s, %s>' % (arg['ir_res'], class_info, using_info)
      return 'OutArg<%d, %s, %s>' % (arg['ir_res'], class_info, using_info)
    return 'TmpArg<%s, %s>' % (class_info, using_info)
  if arg['usage'] == 'use_def':
    if 'ir_res' in arg:
      if need_tmp:
        return 'InOutTmpArg<%s, %s, %s, %s>' % (arg['ir_arg'], arg['ir_res'],
                                                class_info, using_info)
      return 'InOutArg<%s, %s, %s, %s>' % (arg['ir_arg'], arg['ir_res'],
                                           class_info, using_info)
    return 'InTmpArg<%s, %s, %s>' % (arg['ir_arg'], class_info, using_info)
  assert False, 'unknown operand usage %s' % (arg['usage'])


def _gen_make_intrinsics(f, intrs):
  print("""%s
void MakeIntrinsics(FILE* out) {
  using MacroAssembler = MacroAssembler<TextAssembler>;
  namespace OperandClass = x86::OperandClass;
  std::unique_ptr<GenerateAsmCallBase> asm_call_generators[] = {""" % AUTOGEN, file=f)
  for line in _gen_c_intrinsics_generator(intrs):
    print(line, file=f)
  print("""  };
  GenerateAsmCalls(out, std::forward<decltype(asm_call_generators)>(asm_call_generators));
}""", file=f)


def _gen_c_intrinsics_generator(intrs):
  for name, intr in intrs:
    ins = intr.get('in')
    outs = intr.get('out')
    params = _get_in_params(ins)
    formal_args = ', '.join('%s %s' % (type, param) for type, param in params)
    result_type, _ = _get_result_type(outs)
    if 'asm' not in intr:
      continue
    if 'variants' in intr:
      variants = _get_formats_with_descriptions(intr)
      # Sort by index, to keep order close to what _gen_intrs_enum produces.
      variants = sorted(variants, key=lambda variant: variant[1].index)
      # Collect intr_asms for all versions of intrinsic.
      # Note: not all variants are guaranteed to have asm version!
      # If that happens list of intr_asms for that variant would be empty.
      variants = [(desc, [
          intr_asm for intr_asm in intr['asm'] if fmt in intr_asm['variants']
      ]) for fmt, desc in variants]
      # Print intrinsic generator
      for desc, intr_asms in variants:
        if len(intr_asms) > 0:
          if 'raw' in intr['variants']:
            spec = '%d' % (desc.num_elements)
          else:
            spec = '%s, %d' % (desc.c_type, desc.num_elements)
          for intr_asm in intr_asms:
            for line in _gen_c_intrinsic('%s<%s>' % (name, spec), intr, intr_asm):
              yield line
    else:
      for intr_asm in intr['asm']:
        for line in _gen_c_intrinsic(name, intr, intr_asm):
          yield line


MAX_GENERATED_LINE_LENGTH = 100


def _gen_c_intrinsic(name, intr, asm):
  if not _is_interpreter_compatible_assembler(asm):
    return

  sse_restriction = 'GenerateAsmCallBase::kNoSSERestriction'
  if 'feature' in asm:
    if asm['feature'] == 'AuthenticAMD':
      sse_restriction = 'GenerateAsmCallBase::kIsAuthenticAMD'
    else:
      sse_restriction = 'GenerateAsmCallBase::kHas%s' % asm['feature']

  nan_restriction = 'GenerateAsmCallBase::kNoNansOperation'
  if 'nan' in asm:
    nan_restriction = 'GenerateAsmCallBase::k%sNanOperationsHandling' % asm['nan']

  restriction = [sse_restriction, nan_restriction]

  yield '      std::unique_ptr<GenerateAsmCallBase>('

  def get_c_type_tuple(arguments):
    return 'std::tuple<%s>' % ', '.join(
        _get_c_type(argument) for argument in arguments).replace(
            'Float', 'intrinsics::Float')

  yield '          new GenerateAsmCall<%s>(' % (
    ',\n                              '.join(
        ['true' if _intr_has_side_effects(intr) else 'false'] +
        [get_c_type_tuple(intr['in'])] + [get_c_type_tuple(intr['out'])] +
        [_get_reg_operand_info(arg, 'OperandClass')
         for arg in asm['args']]))

  one_line = '              out, &MacroAssembler::%s, %s)),' % (
      asm['asm'], ', '.join(['"%s"' % name] + restriction))
  if len(one_line) <= MAX_GENERATED_LINE_LENGTH:
    yield one_line
    return

  yield '              out,'
  yield '              &MacroAssembler::%s,' % asm['asm']
  values = ['"%s"' % name] + restriction
  for index, value in enumerate(values):
    if index + 1 == len(values):
      yield '              %s)),' % value
    else:
      yield '              %s,' % value


def _load_intrs_def_files(intrs_def_files):
  result = {}
  for intrs_def in intrs_def_files:
    with open(intrs_def) as intrs:
      result.update(json.load(intrs))
  result.pop('License', None)
  return result


def _load_intrs_arch_def(intrs_defs):
  json_data = []
  for intrs_def in intrs_defs:
    with open(intrs_def) as intrs:
      json_array = json.load(intrs)
      while isinstance(json_array[0], str):
        json_array.pop(0)
      json_data.extend(json_array)
  return json_data


def _load_macro_def(intrs, arch_intrs, insns_def):
  _, insns = asm_defs.load_asm_defs(insns_def)
  insns_map = dict((insn['name'], insn) for insn in insns)
  unprocessed_intrs = []
  for arch_intr in arch_intrs:
    if arch_intr['insn'] in insns_map:
      insn = insns_map[arch_intr['insn']]
      _add_asm_insn(intrs, arch_intr, insn)
    else:
      unprocessed_intrs.append(arch_intr)
  return unprocessed_intrs


def _is_interpreter_compatible_assembler(intr_asm):
  if intr_asm.get('usage', '') == 'translate-only':
    return False
  return True


def _add_asm_insn(intrs, arch_intr, insn):
  name = arch_intr['name']
  # Sanity checks: MacroInstruction could implement few different intrinsics but
  # number of arguments in arch intrinsic and arch-independent intrinsic
  # should match.
  #
  # Note: we allow combining intrinsics with variants and intrinsics without
  # variants (e.g. AbsF32 is combined with VectorAbsoluteFP for F32x2 and F32x4),
  # but don't allow macroinstructions which would handle different set of
  # variants for different intrinsics.

  assert 'variants' not in insn or insn['variants'] == arch_intr['variants']
  assert 'feature' not in insn or insn['feature'] == arch_intr['feature']
  assert 'nan' not in insn or insn['nan'] == arch_intr['nan']
  assert 'usage' not in insn or insn['usage'] == arch_intr['usage']
  assert len(intrs[name]['in']) == len(arch_intr['in'])
  assert len(intrs[name]['out']) == len(arch_intr['out'])

  if 'variants' in arch_intr:
    insn['variants'] = arch_intr['variants']
  if 'feature' in arch_intr:
    insn['feature'] = arch_intr['feature']
  if 'nan' in arch_intr:
    insn['nan'] = arch_intr['nan']
  if 'usage' in arch_intr:
    insn['usage'] = arch_intr['usage']

  for count, in_arg in enumerate(arch_intr['in']):
    # Sanity check: each in argument should only be used once - but if two
    # different intrinsics use them same macroinstruction it could be already
    # defined... yet it must be defined identically.
    assert ('ir_arg' not in insn['args'][in_arg] or
            insn['args'][in_arg]['ir_arg'] == count)
    insn['args'][in_arg]['ir_arg'] = count

  for count, out_arg in enumerate(arch_intr['out']):
    # Sanity check: each out argument should only be used once, too.
    assert ('ir_res' not in insn['args'][out_arg] or
            insn['args'][out_arg]['ir_res'] == count)
    insn['args'][out_arg]['ir_res'] = count

  # Note: one intrinsic could have more than one implementation (e.g.
  # SSE2 vs SSE4.2).
  if 'asm' not in intrs[name]:
    intrs[name]['asm'] = []
  intrs[name]['asm'].append(insn)


def _open_asm_def_files(def_files, arch_def_files, asm_def_files):
  intrs = _load_intrs_def_files(def_files)
  arch_intrs = _load_intrs_arch_def(arch_def_files)
  for macro_def in asm_def_files:
    arch_intrs = _load_macro_def(intrs, arch_intrs, macro_def)
  # Make sure that all intrinsics were found during processing of arch_intrs.
  assert arch_intrs == []
  return sorted(intrs.items())


def main(argv):
  # Usage:
  #   gen_intrinsics.py --public_headers <intrinsics-inl.h>
  #                                      <interpreter_intrinsics_hooks-inl.h>
  #                                      <translator_intrinsics_hooks-inl.h>
  #                                      <mock_semantics_listener_intrinsics_hooks-inl.h>
  #                                      <riscv64_to_x86_64/intrinsic_def.json",
  #                                      ...
  #   gen_intrinsics.py --make_intrinsics_cc <make_intrinsics.cc>
  #                                          <riscv64_to_x86_64/intrinsic_def.json",
  #                                          ...
  #                                          <riscv64_to_x86_64/machine_ir_intrinsic_binding.json>,
  #                                          ...
  #                                          <riscv64_to_x86_64/macro_def.json>,
  #                                          ...

  def open_out_file(name):
    try:
      os.makedirs(os.path.dirname(name))
    except:
      pass
    return open(name, 'w')

  mode = argv[1]
  if mode == '--public_headers':
    intrs = sorted(_load_intrs_def_files(argv[6:]).items())
    _gen_intrinsics_inl_h(open_out_file(argv[2]), intrs)
    _gen_interpreter_intrinsics_hooks_impl_inl_h(open_out_file(argv[3]), intrs)
    _gen_translator_intrinsics_hooks_impl_inl_h(
        open_out_file(argv[4]), intrs)
    _gen_mock_semantics_listener_intrinsics_hooks_impl_inl_h(
        open_out_file(argv[5]), intrs)
  elif mode == '--make_intrinsics_cc':
    def_files_end = 3
    while argv[def_files_end].endswith('intrinsic_def.json'):
      def_files_end += 1
    arch_def_files_end = def_files_end
    while argv[arch_def_files_end].endswith('machine_ir_intrinsic_binding.json'):
      arch_def_files_end += 1
    intrs = _open_asm_def_files(
      argv[3:def_files_end],
      argv[def_files_end:arch_def_files_end],
      argv[arch_def_files_end:])
    _gen_make_intrinsics(open_out_file(argv[2]), intrs)
  else:
    assert False, 'unknown option %s' % (mode)

  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))