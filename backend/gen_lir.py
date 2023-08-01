#!/usr/bin/python
#
# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Generate LIR files out of the definition file.

* Operand usage

Register allocator needs operand usage to learn which operands can share the
same register.

To understand register sharing options, register allocator assumes insn works
in these steps:
- read input operands
- do the job
- write output operands

So, input-output operands should have dedicated registers, while input-only
operands can share registers with output-only operands.

There might be an exception when output-only operand is written before all
input-only operands are read, so its register can't be shared. Such operands
are usually referred as output-only-early-clobber operands.

For register sharing, output-only-early-clobber operand is the same as
input-output operand, but it is unnatural to describe output-only as
input-output, so we use a special keyword for it.

Finally, keywords are:
use - input-only
def - output-only
def_early_clobber - output-only-early-clobber
use_def - input-output

* Scratch operands

Scratch operands are actually output operands - indeed, their original value
is not used and they get some new value after the insn is done. However, they
are usually written before all input operands are read, so it makes sense to
describe scratch operands as output-only-early-clobber.
"""

import asm_defs
import json
import sys


def _is_reg(arg_type):
  return (asm_defs.is_greg(arg_type) or
          asm_defs.is_xreg(arg_type) or
          asm_defs.is_implicit_reg(arg_type))


class Operand(object):
  pass


def _get_reg_operand_info(usage, kind):
  if usage == 'use':
    return '{ &k%s, MachineRegKind::kUse }' % (kind)
  if usage == 'def':
    return '{ &k%s, MachineRegKind::kDef }' % (kind)
  if usage == 'use_def':
    return '{ &k%s, MachineRegKind::kUseDef }' % (kind)
  if usage == 'def_early_clobber':
    return '{ &k%s, MachineRegKind::kDefEarlyClobber }' % (kind)
  assert False, 'unknown operand usage %s' % (usage)


def _make_reg_operand(r, usage, kind):
   op = Operand()
   op.type = 'MachineReg'
   op.name = 'r%d' % (r)
   op.reg_operand_info = _get_reg_operand_info(usage, kind)
   op.initializer = 'SetRegAt(%d, r%d)' % (r, r)
   if asm_defs.is_greg(kind):
     op.asm_arg = 'GetGReg(RegAt(%d))' % (r)
   elif asm_defs.is_xreg(kind):
     op.asm_arg = 'GetXReg(RegAt(%d))' % (r)
   elif asm_defs.is_implicit_reg(kind):
     op.asm_arg = None
   else:
     assert False, 'unknown register kind %s' % (kind)
   return op


def _make_imm_operand(bits):
  op = Operand()
  op.type = 'int%s_t' % (bits)
  op.name = 'imm'
  op.reg_operand_info = None
  op.initializer = 'set_imm(imm)'
  op.asm_arg = 'static_cast<%s>(imm())' % (op.type)
  return op


def _make_scale_operand():
  op = Operand()
  op.type = 'MachineMemOperandScale'
  op.name = 'scale'
  op.reg_operand_info = None
  op.initializer = 'set_scale(scale)'
  op.asm_arg = 'ToScaleFactor(scale())'
  return op


def _make_disp_operand():
  op = Operand()
  op.type = 'uint32_t'
  op.name = 'disp'
  op.reg_operand_info = None
  op.initializer = 'set_disp(disp)'
  op.asm_arg = 'disp()'
  return op


def _make_cond_operand():
  op = Operand()
  op.type = 'Assembler::Condition'
  op.name = 'cond'
  op.reg_operand_info = None
  op.initializer = 'set_cond(cond)'
  op.asm_arg = 'cond()'
  return op


def _make_label_operand():
  op = Operand()
  # We never have both immediate and Label in same insn.
  op.type = 'Label*'
  op.name = 'label'
  op.reg_operand_info = None
  op.initializer = 'set_imm(reinterpret_cast<uintptr_t>(label))'
  op.asm_arg = '*reinterpret_cast<Label*>(imm())'
  return op


def _check_insn_defs(insn):
  seen_imm = False
  seen_memop = False
  seen_disp = False
  for arg in insn.get('args'):
    kind = arg.get('class')
    if _is_reg(kind):
      pass
    elif asm_defs.is_imm(kind):
      # We share field for immediate and label in 'insn'.
      assert not seen_imm
      seen_imm = True
    elif asm_defs.is_mem_op(kind):
      # No insn can have more than one memop.
      assert not seen_memop
      addr_mode = insn.get('addr_mode')
      assert addr_mode in ('Absolute', 'BaseDisp', 'IndexDisp', 'BaseIndexDisp'), \
        'unknown addressing mode %s' % (addr_mode)
      seen_memop = True
    elif asm_defs.is_disp(kind):
      assert not seen_disp
      seen_disp = True
    elif asm_defs.is_cond(kind):
      pass
    elif asm_defs.is_label(kind):
      assert not seen_imm
      seen_imm = True
    else:
      assert False, 'unknown operand class %s' % (kind)


def _get_insn_operands(insn):
  """For each operand, define:
  - type
  - name
  - reg_operand_info
  - initializer
  - asm_arg
  """
  res = []
  r = 0
  # Int3, Mfence, and UD2 have side effects not related to arguments.
  side_effects = insn['name'] in ('Int3', 'Mfence', 'UD2')
  for arg in insn.get('args'):
    kind = arg.get('class')
    if _is_reg(kind):
      res.append(_make_reg_operand(r, arg.get('usage'), kind))
      r += 1
    elif asm_defs.is_imm(kind):
      # We share field for immediate and label in 'insn'.
      bits = kind[3:]
      res.append(_make_imm_operand(bits))
    elif asm_defs.is_mem_op(kind):
      # If operand is memory and it's not "use" then we have side_effects
      if arg['usage'] != 'use':
        side_effects = True
      # No insn can have more than one memop.
      addr_mode = insn.get('addr_mode')
      assert addr_mode in ('Absolute', 'BaseDisp', 'IndexDisp', 'BaseIndexDisp'), \
        'unknown addressing mode %s' % (addr_mode)
      if addr_mode in ('BaseDisp', 'BaseIndexDisp'):
        res.append(_make_reg_operand(r, 'use', 'GeneralReg32'))
        r += 1

      if addr_mode in ('IndexDisp', 'BaseIndexDisp'):
        res.append(_make_reg_operand(r, 'use', 'GeneralReg32'))
        r += 1
        res.append(_make_scale_operand())

      res.append(_make_disp_operand())
    elif asm_defs.is_disp(kind):
      res.append(_make_disp_operand())
    elif asm_defs.is_cond(kind):
      res.append(_make_cond_operand())
    elif asm_defs.is_label(kind):
      res.append(_make_label_operand())
    else:
      assert False, 'unknown operand class %s' % (kind)
  return res, side_effects


def _get_insn_debug_operands(insn):
  res = []
  r = 0
  for arg in insn.get('args'):
    kind = arg.get('class')
    if _is_reg(kind):
      if asm_defs.is_greg(kind) or asm_defs.is_xreg(kind):
        res.append('GetRegOperandDebugString(this, %d)' % (r))
      elif asm_defs.is_implicit_reg(kind):
        res.append('GetImplicitRegOperandDebugString(this, %d)' % (r))
      else:
        assert False, 'unknown register kind %s' % (kind)
      r += 1
    elif asm_defs.is_imm(kind):
      # We share field for immediate and label in 'insn'.
      res.append('GetImmOperandDebugString(this)')
    elif asm_defs.is_mem_op(kind):
      # No insn can have more than one memop.
      addr_mode = insn.get('addr_mode')
      if addr_mode == 'Absolute':
        res.append('GetAbsoluteMemOperandDebugString(this)')
      elif addr_mode in ('BaseDisp', 'IndexDisp', 'BaseIndexDisp'):
        res.append('Get%sMemOperandDebugString(this, %d)' % (addr_mode, r))
        r += {'BaseDisp': 1, 'IndexDisp': 1, 'BaseIndexDisp': 2}[addr_mode]
      else:
        assert False, 'unknown addr_mode %s' % (addr_mode)
    elif asm_defs.is_disp(kind):
      # Hack: replace previous reg helper with mem helper.
      assert res
      assert res[-1].startswith('GetRegOperandDebugString')
      res[-1] = 'GetBaseDispMemOperandDebugString' + res[-1][24:]
    elif asm_defs.is_cond(kind):
      res.append('GetCondOperandDebugString(this)')
    elif asm_defs.is_label(kind):
      res.append('GetLabelOperandDebugString(this)')
    else:
      assert False, 'unknown operand class %s' % (kind)
  return res


INDENT = '  '


def _gen_insn_ctor(f, insn):
  name = insn.get('name')
  operands, _ = _get_insn_operands(insn)
  params = ['%s %s' % (op.type, op.name) for op in operands]
  inits = ['%s%s;' % (INDENT, op.initializer) for op in operands]
  print('constexpr MachineInsnInfo %s::kInfo;' % (name), file=f)
  print('%s::%s(%s) : MachineInsnForArch(&kInfo) {' % (name, name, ', '.join(params)), file=f)
  print('\n'.join(inits), file=f)
  print('}', file=f)


# TODO(ndk_translation-eng): Maybe we should just implement generic printing in C++
# instead of generating it for every instruction.
def _gen_insn_debug(f, insn):
  name = insn.get('name')
  mnemo = insn.get('mnemo')
  print('string %s::GetDebugString() const {' % (name), file=f)
  operands = _get_insn_debug_operands(insn)
  if not operands:
    print('  return "%s";' % (mnemo), file=f)
  else:
    print('  string s("%s ");' % (mnemo), file=f)
    print('  s += %s;' % (operands[0]), file=f)
    for op in operands[1:]:
      print('  s += ", ";', file=f)
      print('  s += %s;' % (op), file=f)
    # We don't print recovery_bb() since it can be found by edges outgoing from basic block.
    print('  if (recovery_pc()) {', file=f)
    print('    s += StringPrintf(" <0x%" PRIxPTR ">", recovery_pc());', file=f)
    print('  }', file=f)
    print('  return s;', file=f)
  print('}', file=f)


def _gen_insn_emit(f, insn):
  name = insn.get('name')
  asm = insn.get('asm')
  operands, _ = _get_insn_operands(insn)
  asm_args = [op.asm_arg for op in operands if op.asm_arg]
  print('void %s::Emit(CodeEmitter* as) const {' % (name), file=f)
  print('%sas->%s(%s);' % (INDENT, asm, ', '.join(asm_args)), file=f)
  print('}', file=f)


def _gen_insn_class(f, insn):
  name = insn.get('name')
  operands, side_effects = _get_insn_operands(insn)
  regs = [op.reg_operand_info for op in operands if op.reg_operand_info]
  if side_effects:
    kind = 'kMachineInsnSideEffects'
  else:
    kind = 'kMachineInsnDefault'
  params = ['%s %s' % (op.type, op.name) for op in operands]
  print('class %s : public MachineInsnForArch {' % (name), file=f)
  print(' public:', file=f)
  print('  explicit %s(%s);' % (name, ', '.join(params)), file=f)
  print('  static constexpr MachineInsnInfo kInfo =', file=f)
  print('      MachineInsnInfo({kMachineOp%s,' % (name), file=f)
  print('                       %d,' % (len(regs)), file=f)
  print('                       {%s},' % (', '.join(regs)), file=f)
  print('                       %s});' % (kind), file=f)
  print('  static constexpr int NumRegOperands() { return kInfo.num_reg_operands; }', file=f)
  print('  static constexpr const MachineRegKind& RegKindAt(int i) { return kInfo.reg_kinds[i]; }', file=f)
  print('  std::string GetDebugString() const override;', file=f)
  print('  void Emit(CodeEmitter* as) const override;', file=f)
  print('};', file=f)


def _gen_code_2_cc(out, arch, insns):
  with open(out, 'w') as f:
    for insn in insns:
      _gen_insn_ctor(f, insn)


def _gen_code_debug_cc(out, arch, insns):
  with open(out, 'w') as f:
    print("""\
// This file automatically generated by gen_lir.py
// DO NOT EDIT!

#include "ndk_translation/base/stringprintf.h"

#include "%s/code_debug.h"

namespace ndk_translation {

namespace %s {
""" % (arch, arch), file=f)
    for insn in insns:
      _gen_insn_debug(f, insn)
    print("""\

}  // namespace %s

}  // namespace ndk_translation""" % (arch), file=f)


def _gen_code_emit_cc(out, arch, insns):
  with open(out, 'w') as f:
    print("""\
// This file automatically generated by gen_lir.py
// DO NOT EDIT!

#include "%s/code_emit.h"

namespace ndk_translation {

namespace %s {
""" % (arch, arch), file=f)
    for insn in insns:
      _gen_insn_emit(f, insn)
    print("""\

}  // namespace %s

}  // namespace ndk_translation""" % (arch), file=f)


def _gen_machine_info_h(out, arch, insns):
  with open(out, 'w') as f:
    for insn in insns:
      name = insn.get('name')
      print('typedef %s %s;' % (name, name), file=f)


def _gen_machine_opcode_h(out, arch, insns):
  with open(out, 'w') as f:
    for insn in insns:
      name = insn.get('name')
      print('kMachineOp%s,' % (name), file=f)


def _gen_mem_insn_groups(f, insns):
  # Build a dictionary to map a memory insn group name to another dictionary,
  # which in turn maps an addressing mode to an individual memory insn.
  groups = {}
  for i in insns:
    group_name = i.get('mem_group_name')
    if group_name:
      groups.setdefault(group_name, {})[i.get('addr_mode')] = i.get('name')

  for group_name in sorted(groups):
    # The order of the addressing modes here is important.  It must
    # match what MemInsns expects.
    mem_insns = [groups[group_name][addr_mode]
                 for addr_mode in ('Absolute', 'BaseDisp', 'IndexDisp', 'BaseIndexDisp')]
    print('typedef MemInsns<%s> %s;' % (', '.join(mem_insns), group_name), file=f)


def _gen_machine_ir_h(out, arch, insns):
  with open(out, 'w') as f:
    for insn in insns:
      _gen_insn_class(f, insn)
    print('', file=f)
    _gen_mem_insn_groups(f, insns)


def _contains_mem(insn):
  return any(asm_defs.is_mem_op(arg['class']) for arg in insn.get('args'))


def _create_mem_insn(insn, addr_mode):
  new_insn = insn.copy()
  macro_name = asm_defs.get_mem_macro_name(insn, addr_mode)
  new_insn['name'] = macro_name
  new_insn['addr_mode'] = addr_mode
  new_insn['asm'] = macro_name
  new_insn['mem_group_name'] = asm_defs.get_mem_macro_name(insn, '') + 'Insns'
  return new_insn


def _expand_mem_insns(insns):
  result = []
  for insn in insns:
    if _contains_mem(insn):
      result.extend([_create_mem_insn(insn, addr_mode)
                     for addr_mode in ('Absolute', 'BaseDisp', 'IndexDisp', 'BaseIndexDisp')])
    else:
      result.append(insn)
  return result


def _load_lir_def(whitelist_looked, whitelist_found, asm_def):
  arch, insns = asm_defs.load_asm_defs(asm_def)
  insns = _expand_mem_insns(insns)
  # Mark all instructions to remove and remember instructions we kept
  for insn in insns:
    insn_name = insn.get('mem_group_name', insn['name'])
    if insn_name in whitelist_looked:
      whitelist_found.add(insn_name)
    else:
      insn['skip_lir'] = 1
  # Filter out disabled instructions.
  insns = [i for i in insns if not i.get('skip_lir')]
  return arch, insns


def _whitelist_instructions(whitelist_file, machine_ir_intrinsic_binding_files):
  whitelisted_names = set()
  with open(whitelist_file) as whitelist_json:
    for insn_name in json.load(whitelist_json):
      whitelisted_names.add(insn_name)
  for machine_ir_intrinsic_binding_file in machine_ir_intrinsic_binding_files:
    with open(machine_ir_intrinsic_binding_file) as machine_ir_intrinsic_binding_json:
      for insn in json.load(machine_ir_intrinsic_binding_json):
        if insn.get('usage', '') != 'interpret-only':
          whitelisted_names.add(insn['insn'])
  return whitelisted_names


def main(argv):
  # Usage:
  #   gen_lir.py --headers <insn-inl.h>
  #                        <machine_info-inl.h>
  #                        <machine_opcode-inl.h>
  #                        <machine_ir-inl.h>
  #                        <lir_instructions.json>
  #                        <machine_ir_intrinsic_binding_x86_common.json>
  #                        <machine_ir_intrinsic_binding_x86.json>
  #                        <def> ... <def>
  #   gen_lir.py --sources <code_emit.cc>
  #                        <code_debug.cc>
  #                        <lir_instructions.json>
  #                        <machine_ir_intrinsic_binding_x86_common.json>
  #                        <machine_ir_intrinsic_binding_x86.json>
  #                        <def> ... <def>

  def load_all_lir_defs(whitelist_file, machine_ir_intrinsic_binding_files, lir_defs):
    whitelist_looked = _whitelist_instructions(
        whitelist_file, machine_ir_intrinsic_binding_files)
    whitelist_found = set()
    arch = None
    insns = []
    for lir_def in lir_defs:
      def_arch, def_insns = _load_lir_def(whitelist_looked, whitelist_found, lir_def)
      if arch and not arch.startswith('common_'):
        assert def_arch is None or arch == def_arch
      else:
        arch = def_arch
      insns.extend(def_insns)
    for i in insns:
      _check_insn_defs(i)
    assert whitelist_looked == whitelist_found
    return arch, insns

  mode = argv[1]
  if mode == '--headers':
    arch, insns = load_all_lir_defs(argv[6], argv[7:9], argv[9:])
    _gen_code_2_cc(argv[2], arch, insns)
    _gen_machine_info_h(argv[3], arch, insns)
    _gen_machine_opcode_h(argv[4], arch, insns)
    _gen_machine_ir_h(argv[5], arch, insns)
  elif mode == '--sources':
    arch, insns = load_all_lir_defs(argv[4], argv[5:7], argv[7:])
    _gen_code_emit_cc(argv[2], arch, insns)
    _gen_code_debug_cc(argv[3], arch, insns)
  else:
    assert False, 'unknown option %s' % (mode)

  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
