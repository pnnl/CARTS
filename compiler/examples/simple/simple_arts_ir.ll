; ModuleID = 'simple_arts_ir.bc'
source_filename = "simple.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str = private unnamed_addr constant [36 x i8] c"EDT 1: The initial number is %d/%d\0A\00", align 1
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@.str.1 = private unnamed_addr constant [28 x i8] c"EDT 0: The number is %d/%d\0A\00", align 1
@.str.3 = private unnamed_addr constant [28 x i8] c"EDT 3: The number is %d/%d\0A\00", align 1
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 322, i32 0, i32 22, ptr @0 }, align 8
@.str.4 = private unnamed_addr constant [37 x i8] c"EDT 2: The final number is %d - %d.\0A\00", align 1

; Function Attrs: mustprogress norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
entry:
  call void @carts.edt.main()
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local void @srand(i32 noundef) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local i64 @time(ptr noundef) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nounwind
declare dso_local i32 @rand() local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #3

; Function Attrs: nocallback nofree norecurse nosync nounwind willreturn memory(readwrite)
define internal void @carts.edt.parallel(ptr nocapture %0, ptr nocapture readonly %1) #4 !carts !4 {
pre_entry:
  br label %entry

entry:                                            ; preds = %pre_entry
  %2 = load i32, ptr %0, align 4, !tbaa !6
  %3 = load i32, ptr %1, align 4, !tbaa !6
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %2, i32 noundef %3) #6
  %5 = load i32, ptr %1, align 4, !tbaa !6
  call void @carts.edt.task(ptr nocapture %0, i32 %5) #9
  br label %exit

exit:                                             ; preds = %entry
  ret void
}

; Function Attrs: convergent nounwind
declare i32 @__kmpc_single(ptr, i32) local_unnamed_addr #5

; Function Attrs: convergent nounwind
declare void @__kmpc_end_single(ptr, i32) local_unnamed_addr #5

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nocallback nofree norecurse nosync nounwind willreturn memory(readwrite)
define internal void @carts.edt.task(ptr nocapture %0, i32 %1) #4 !carts !10 {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !12)
  %3 = load i32, ptr %0, align 4, !tbaa !6, !noalias !12
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %0, align 4, !tbaa !6, !noalias !12
  %5 = add nsw i32 %1, 1
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %4, i32 noundef %5) #6, !noalias !12
  ret void
}

; Function Attrs: nounwind
declare ptr @__kmpc_omp_task_alloc(ptr, i32, i32, i64, i64, ptr) local_unnamed_addr #6

; Function Attrs: nounwind
declare i32 @__kmpc_omp_task(ptr, i32, ptr) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) local_unnamed_addr #5

; Function Attrs: nounwind
declare !callback !15 void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #7

; Function Attrs: norecurse nounwind
define internal void @carts.edt.main() #8 !carts !17 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = tail call i64 @time(ptr noundef null) #6
  %4 = trunc i64 %3 to i32
  tail call void @srand(i32 noundef %4) #6
  %5 = tail call i32 @rand() #6
  %6 = srem i32 %5, 100
  %7 = add nsw i32 %6, 1
  store i32 %7, ptr %1, align 4, !tbaa !6
  %8 = tail call i32 @rand() #6
  %9 = srem i32 %8, 10
  %10 = add nsw i32 %9, 1
  store i32 %10, ptr %2, align 4, !tbaa !6
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %7, i32 noundef %10) #6
  call void @carts.edt.parallel(ptr nocapture %1, ptr nocapture %2) #10
  br label %codeRepl

codeRepl:                                         ; preds = %0
  call void @carts.edt.sync.done(ptr nocapture %1, ptr nocapture %2) #6
  br label %.split.ret

.split.ret:                                       ; preds = %codeRepl
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @carts.edt.sync.done(ptr nocapture %0, ptr nocapture readonly %1) #8 !carts !19 {
newFuncRoot:
  br label %.split

.split:                                           ; preds = %newFuncRoot
  %2 = load i32, ptr %0, align 4, !tbaa !6
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr %0, align 4, !tbaa !6
  %4 = load i32, ptr %1, align 4, !tbaa !6
  %5 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %3, i32 noundef %4) #6
  br label %.split.ret.exitStub

.split.ret.exitStub:                              ; preds = %.split
  ret void
}

attributes #0 = { mustprogress norecurse nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nocallback nofree norecurse nosync nounwind willreturn memory(readwrite) }
attributes #5 = { convergent nounwind }
attributes #6 = { nounwind }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #8 = { norecurse nounwind }
attributes #9 = { memory(argmem: readwrite) }
attributes #10 = { nounwind memory(readwrite) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"openmp", i32 50}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 16.0.6 (Red Hat 16.0.6-1.el9)"}
!4 = !{!"sync", !5}
!5 = !{!"dep", !"dep"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!"task", !11}
!11 = !{!"dep", !"param"}
!12 = !{!13}
!13 = distinct !{!13, !14, !".omp_outlined..2: argument 0"}
!14 = distinct !{!14, !".omp_outlined..2"}
!15 = !{!16}
!16 = !{i64 2, i64 -1, i64 -1, i1 true}
!17 = !{!"main", !18}
!18 = !{}
!19 = !{!"task", !5}
