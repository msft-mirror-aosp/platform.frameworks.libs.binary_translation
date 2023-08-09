#!/usr/bin/python3
#
# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Generate machine IR register classes definition from data file."""

import json
import sys


def _gen_machine_reg_class_inc(f, reg_classes):
  for reg_class in reg_classes:
    name = reg_class.get('name')
    regs = reg_class.get('regs')
    print('inline constexpr uint64_t k%sMask =' % (name), file=f)
    for r in regs[: -1]:
      print('    (1ULL << kMachineReg%s.reg()) |' % (r), file=f)
    print('    (1ULL << kMachineReg%s.reg());' % (regs[-1]), file=f)
    print('inline constexpr ndk_translation::MachineRegClass k%s = {' % (name), file=f)
    print('    "%s",' % (name), file=f)
    print('    %d,' % (reg_class.get('size')), file=f)
    print('    k%sMask,' % (name), file=f)
    print('    %d,' % (len(regs)), file=f)
    print('    {', file=f)
    for r in regs:
      print('      kMachineReg%s,' % (r), file=f)
    print('    }', file=f)
    print('};', file=f)


def _expand_aliases(reg_classes):
  expanded = {}
  for reg_class in reg_classes:
    expanded_regs = []
    for r in reg_class.get('regs'):
      expanded_regs.extend(expanded.get(r, [r]))
    reg_class['regs'] = expanded_regs
    expanded[reg_class.get('name')] = expanded_regs


def main(argv):
  # Usage: gen_reg_class.py <machine_reg_class-inl.h> <def> ... <def>
  machine_reg_class_inl_name = argv[1]
  defs = argv[2:]

  reg_classes = []
  for d in defs:
    f = open(d)
    j = json.load(f)
    reg_classes.extend(j.get('reg_classes'))

  _expand_aliases(reg_classes)

  _gen_machine_reg_class_inc(open(machine_reg_class_inl_name, 'w'), reg_classes)


if __name__ == '__main__':
  sys.exit(main(sys.argv))
