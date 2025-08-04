///===========================================================================
/// EDTInvariantCodeMotion
///===========================================================================

/// Dialects
#include "arts/Utils/ArtsUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <queue>

/// Arts
#include "arts/ArtsDialect.h"
#include "arts/Transforms/EdtInvariantCodeMotion.h"

/// Others
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"

/// LLVM support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "edt-invariant-code-motion"
#define LINE "-----------------------------------------\n"
#define dbgs() (llvm::dbgs())
#define DBGS() (dbgs() << "[" DEBUG_TYPE "] ")

namespace mlir {
namespace arts {

static inline bool isDefinedOutsideEdt(Region &region, Value value) {
  return !region.isAncestor(value.getParentRegion());
};

bool canBeHoistedFromEdt(Region &edtRegion, Operation *op) {
  /// Do not move terminators.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;

  /// Do not move other ARTS ops
  if (isArtsOp(op))
    return false;

  /// Do not move pointers or memref operations
  for (Value result : op->getResults()) {
    if (result.getType().isa<MemRefType>() ||
        result.getType().isa<LLVM::LLVMPointerType>())
      return false;
  }

  /// If it is memory effect free and speculatable, it is invariant.
  if (!mlir::isPure(op))
    return false;

  /// Walk the nested operations and check that all used values are defined
  /// outside of the EDT region
  auto walkFn = [&](Operation *child) {
    for (Value operand : child->getOperands()) {
      /// Ignore values defined in a nested region
      if (op->isAncestor(operand.getParentRegion()->getParentOp()))
        continue;
      if (!isDefinedOutsideEdt(edtRegion, operand))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  return !op->walk(walkFn).wasInterrupted();
}

uint64_t moveEdtInvariantCode(EdtOp edtOp) {
  uint64_t numMoved = 0;
  Region &region = edtOp.getRegion();

  /// Define the callback to actually move the operation.
  auto moveOutOfEdt = [&](Operation *op) { op->moveBefore(edtOp); };

  std::queue<Operation *> worklist;
  /// Add top-level operations in the EDT body to the worklist.
  for (Operation &op : region.getOps())
    worklist.push(&op);

  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop();

    /// Skip ops that have already been moved or processed
    if (op->getParentRegion() != &region)
      continue;

    /// Check if the op should be moved and can be hoisted
    LLVM_DEBUG(dbgs() << "Checking op:\n"
                      << "  - " << *op << "\n");
    if (!canBeHoistedFromEdt(region, op))
      continue;

    LLVM_DEBUG(dbgs() << "  - Moving EDT-invariant op\n");
    moveOutOfEdt(op);
    ++numMoved;

    /// Since the op has been moved, we need to check its users within the
    /// top-level of the edt region.
    for (Operation *user : op->getUsers())
      if (user->getParentRegion() == &region)
        worklist.push(user);
  }

  return numMoved;
}

} // namespace arts
} // namespace mlir
