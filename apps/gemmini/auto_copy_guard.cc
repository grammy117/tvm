/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_copy_guard.cc
 * \brief Explicit host-only validation when automatic copy lowering is unavailable.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>

namespace tvm {
namespace tir {
namespace {

class HostOnlyAutoCopyGuard : public StmtVisitor {
 public:
  static void CheckKey(const String& key) {
    ICHECK(std::string(key).find("auto_copy") == std::string::npos) << "HostOnlyAutoCopyGuard: annotation '" << key << "' requires automatic copy lowering, which is unavailable in this host-only build";
  }

  static void CheckAnnotations(const Map<String, ObjectRef>& annotations) {
    for (const auto& entry : annotations) {
      CheckKey(entry.first);
    }
  }

 private:
  void VisitStmt_(const BlockNode* op) final {
    CheckAnnotations(op->annotations);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    CheckAnnotations(op->annotations);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    CheckKey(op->attr_key);
    StmtVisitor::VisitStmt_(op);
  }
};

}  // namespace

namespace transform {

Pass LowerAutoCopy() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    if (func->attrs.defined()) {
      HostOnlyAutoCopyGuard::CheckAnnotations(func->attrs->dict);
    }
    HostOnlyAutoCopyGuard visitor;
    visitor(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.HostOnlyAutoCopyGuard", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAutoCopy").set_body_typed(LowerAutoCopy);
TVM_REGISTER_GLOBAL("tir.transform.HostOnlyAutoCopyGuardEnabled").set_body_typed([]() { return true; });

}  // namespace transform
}  // namespace tir
}  // namespace tvm
