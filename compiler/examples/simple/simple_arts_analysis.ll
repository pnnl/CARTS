; ModuleID = 'simple_arts_analysis.bc'
source_filename = "simple.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.artsEdtDep_t = type { i64, i32, ptr }

@.str = private unnamed_addr constant [36 x i8] c"EDT 1: The initial number is %d/%d\0A\00", align 1
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@.str.1 = private unnamed_addr constant [28 x i8] c"EDT 0: The number is %d/%d\0A\00", align 1
@.str.3 = private unnamed_addr constant [28 x i8] c"EDT 3: The number is %d/%d\0A\00", align 1
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 322, i32 0, i32 22, ptr @0 }, align 8
@.str.4 = private unnamed_addr constant [37 x i8] c"EDT 2: The final number is %d - %d.\0A\00", align 1

; Function Attrs: nounwind
declare dso_local void @srand(i32 noundef) local_unnamed_addr #0

; Function Attrs: nounwind
declare dso_local i64 @time(ptr noundef) local_unnamed_addr #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind
declare dso_local i32 @rand() local_unnamed_addr #0

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare i32 @__kmpc_single(ptr, i32) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare void @__kmpc_end_single(ptr, i32) local_unnamed_addr #3

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare ptr @__kmpc_omp_task_alloc(ptr, i32, i32, i64, i64, ptr) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @__kmpc_omp_task(ptr, i32, ptr) local_unnamed_addr #4

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) local_unnamed_addr #3

; Function Attrs: nounwind
declare !callback !4 void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #5

define internal void @edt_1.task(i32 %paramc, ptr %paramv, i32 %depc, ptr %depv) {
entry:
  %edt_1.paramv_0 = getelementptr inbounds i64, ptr %paramv, i64 0
  %0 = load i64, ptr %edt_1.paramv_0, align 8
  %1 = trunc i64 %0 to i32
  %edt_1.paramv_1.guid.edt_3 = getelementptr inbounds i64, ptr %paramv, i64 1
  %2 = load i64, ptr %edt_1.paramv_1.guid.edt_3, align 8
  %edt_1.depv_0.guid = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 0, i32 0
  %edt_1.depv_0.guid.load = load i64, ptr %edt_1.depv_0.guid, align 8
  %edt_1.depv_0.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 0, i32 2
  %edt_1.depv_0.ptr.load = load ptr, ptr %edt_1.depv_0.ptr, align 8
  br label %edt.body

edt.body:                                         ; preds = %entry
  tail call void @llvm.experimental.noalias.scope.decl(metadata !6)
  %3 = load i32, ptr %edt_1.depv_0.ptr.load, align 4, !tbaa !9, !noalias !6
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %edt_1.depv_0.ptr.load, align 4, !tbaa !9, !noalias !6
  %5 = add nsw i32 %1, 1
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %4, i32 noundef %5) #4, !noalias !6
  br label %exit

exit:                                             ; preds = %edt.body
  %toedt.3.slot.0 = alloca i32, align 4
  store i32 0, ptr %toedt.3.slot.0, align 4
  %7 = load i32, ptr %toedt.3.slot.0, align 4
  call void @artsSignalEdt(i64 %2, i32 %7, i64 %edt_1.depv_0.guid.load)
  ret void
}

define internal void @edt_0.sync(i32 %paramc, ptr %paramv, i32 %depc, ptr %depv) {
entry:
  %0 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %edt_1.task_guid.addr = alloca i64, align 8
  store i64 %0, ptr %edt_1.task_guid.addr, align 8
  %1 = load i64, ptr %edt_1.task_guid.addr, align 8
  %edt_0.paramv_0.guid.edt_3 = getelementptr inbounds i64, ptr %paramv, i64 0
  %2 = load i64, ptr %edt_0.paramv_0.guid.edt_3, align 8
  %edt_0.depv_0.guid = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 0, i32 0
  %edt_0.depv_0.guid.load = load i64, ptr %edt_0.depv_0.guid, align 8
  %edt_0.depv_0.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 0, i32 2
  %edt_0.depv_0.ptr.load = load ptr, ptr %edt_0.depv_0.ptr, align 8
  %edt_0.depv_1.guid = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 1, i32 0
  %edt_0.depv_1.guid.load = load i64, ptr %edt_0.depv_1.guid, align 8
  %edt_0.depv_1.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 1, i32 2
  %edt_0.depv_1.ptr.load = load ptr, ptr %edt_0.depv_1.ptr, align 8
  br label %edt.body

edt.body:                                         ; preds = %entry
  br label %entry1

entry1:                                           ; preds = %edt.body
  %3 = load i32, ptr %edt_0.depv_0.ptr.load, align 4, !tbaa !9
  %4 = load i32, ptr %edt_0.depv_1.ptr.load, align 4, !tbaa !9
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %3, i32 noundef %4) #4
  %6 = load i32, ptr %edt_0.depv_1.ptr.load, align 4, !tbaa !9
  %edt_1_paramc = alloca i32, align 4
  store i32 2, ptr %edt_1_paramc, align 4
  %7 = load i32, ptr %edt_1_paramc, align 4
  %edt_1_paramv = alloca i64, i32 %7, align 8
  %edt_1.paramv_0 = getelementptr inbounds i64, ptr %edt_1_paramv, i64 0
  %8 = sext i32 %6 to i64
  store i64 %8, ptr %edt_1.paramv_0, align 8
  %edt_1.paramv_1.guid.edt_3 = getelementptr inbounds i64, ptr %edt_1_paramv, i64 1
  store i64 %2, ptr %edt_1.paramv_1.guid.edt_3, align 8
  %9 = alloca i32, align 4
  store i32 1, ptr %9, align 4
  %10 = load i32, ptr %9, align 4
  %11 = call i64 @artsEdtCreateWithGuid(ptr @edt_1.task, i64 %1, i32 %7, ptr %edt_1_paramv, i32 %10)
  br label %exit2

exit2:                                            ; preds = %entry1
  br label %exit

exit:                                             ; preds = %exit2
  %toedt.1.slot.0 = alloca i32, align 4
  store i32 0, ptr %toedt.1.slot.0, align 4
  %12 = load i32, ptr %toedt.1.slot.0, align 4
  call void @artsSignalEdt(i64 %1, i32 %12, i64 %edt_0.depv_0.guid.load)
  %toedt.3.slot.1 = alloca i32, align 4
  store i32 1, ptr %toedt.3.slot.1, align 4
  %13 = load i32, ptr %toedt.3.slot.1, align 4
  call void @artsSignalEdt(i64 %2, i32 %13, i64 %edt_0.depv_1.guid.load)
  ret void
}

declare i64 @artsReserveGuidRoute(i32, i32)

define internal void @edt_3.task(i32 %paramc, ptr %paramv, i32 %depc, ptr %depv) {
entry:
  %edt_3.depv_0.guid = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 0, i32 0
  %edt_3.depv_0.guid.load = load i64, ptr %edt_3.depv_0.guid, align 8
  %edt_3.depv_0.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 0, i32 2
  %edt_3.depv_0.ptr.load = load ptr, ptr %edt_3.depv_0.ptr, align 8
  %edt_3.depv_1.guid = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 1, i32 0
  %edt_3.depv_1.guid.load = load i64, ptr %edt_3.depv_1.guid, align 8
  %edt_3.depv_1.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i32 1, i32 2
  %edt_3.depv_1.ptr.load = load ptr, ptr %edt_3.depv_1.ptr, align 8
  br label %edt.body

edt.body:                                         ; preds = %entry
  br label %.split

.split:                                           ; preds = %edt.body
  %0 = load i32, ptr %edt_3.depv_0.ptr.load, align 4, !tbaa !9
  %1 = add nsw i32 %0, 1
  store i32 %1, ptr %edt_3.depv_0.ptr.load, align 4, !tbaa !9
  %2 = load i32, ptr %edt_3.depv_1.ptr.load, align 4, !tbaa !9
  %3 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %1, i32 noundef %2) #4
  br label %.split.ret.exitStub

.split.ret.exitStub:                              ; preds = %.split
  br label %exit

exit:                                             ; preds = %.split.ret.exitStub
  call void @artsShutdown()
  ret void
}

define internal void @edt_2.main(i32 %paramc, ptr %paramv, i32 %depc, ptr %depv) {
entry:
  %0 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %edt_3.task_guid.addr = alloca i64, align 8
  store i64 %0, ptr %edt_3.task_guid.addr, align 8
  %1 = load i64, ptr %edt_3.task_guid.addr, align 8
  %2 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %edt_0.sync_guid.addr = alloca i64, align 8
  store i64 %2, ptr %edt_0.sync_guid.addr, align 8
  %3 = load i64, ptr %edt_0.sync_guid.addr, align 8
  %db.0.addr = alloca i64, align 8
  %db.0.size = alloca i64, align 8
  store i64 4, ptr %db.0.size, align 8
  %db.0.size.ld = load i64, ptr %db.0.size, align 8
  %4 = call i64 @artsDbCreatePtr(ptr %db.0.addr, i64 %db.0.size.ld, i32 7)
  %db.0.addr.ld = load i64, ptr %db.0.addr, align 8
  %db.0.ptr = inttoptr i64 %db.0.addr.ld to ptr
  %db.1.addr = alloca i64, align 8
  %db.1.size = alloca i64, align 8
  store i64 4, ptr %db.1.size, align 8
  %db.1.size.ld = load i64, ptr %db.1.size, align 8
  %5 = call i64 @artsDbCreatePtr(ptr %db.1.addr, i64 %db.1.size.ld, i32 7)
  %db.1.addr.ld = load i64, ptr %db.1.addr, align 8
  %db.1.ptr = inttoptr i64 %db.1.addr.ld to ptr
  br label %edt.body

edt.body:                                         ; preds = %entry
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = tail call i64 @time(ptr noundef null) #4
  %9 = trunc i64 %8 to i32
  tail call void @srand(i32 noundef %9) #4
  %10 = tail call i32 @rand() #4
  %11 = srem i32 %10, 100
  %12 = add nsw i32 %11, 1
  store i32 %12, ptr %db.0.ptr, align 4, !tbaa !9
  %13 = tail call i32 @rand() #4
  %14 = srem i32 %13, 10
  %15 = add nsw i32 %14, 1
  store i32 %15, ptr %db.1.ptr, align 4, !tbaa !9
  %16 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %12, i32 noundef %15) #4
  %edt_0_paramc = alloca i32, align 4
  store i32 1, ptr %edt_0_paramc, align 4
  %17 = load i32, ptr %edt_0_paramc, align 4
  %edt_0_paramv = alloca i64, i32 %17, align 8
  %edt_0.paramv_0.guid.edt_3 = getelementptr inbounds i64, ptr %edt_0_paramv, i64 0
  store i64 %1, ptr %edt_0.paramv_0.guid.edt_3, align 8
  %18 = alloca i32, align 4
  store i32 2, ptr %18, align 4
  %19 = load i32, ptr %18, align 4
  %20 = call i64 @artsEdtCreateWithGuid(ptr @edt_0.sync, i64 %3, i32 %17, ptr %edt_0_paramv, i32 %19)
  br label %codeRepl

codeRepl:                                         ; preds = %edt.body
  %edt_3_paramc = alloca i32, align 4
  store i32 0, ptr %edt_3_paramc, align 4
  %21 = load i32, ptr %edt_3_paramc, align 4
  %edt_3_paramv = alloca i64, i32 %21, align 8
  %22 = alloca i32, align 4
  store i32 2, ptr %22, align 4
  %23 = load i32, ptr %22, align 4
  %24 = call i64 @artsEdtCreateWithGuid(ptr @edt_3.task, i64 %1, i32 %21, ptr %edt_3_paramv, i32 %23)
  br label %.split.ret

.split.ret:                                       ; preds = %codeRepl
  br label %exit

exit:                                             ; preds = %.split.ret
  %toedt.0.slot.0 = alloca i32, align 4
  store i32 0, ptr %toedt.0.slot.0, align 4
  %25 = load i32, ptr %toedt.0.slot.0, align 4
  call void @artsSignalEdt(i64 %3, i32 %25, i64 %4)
  %toedt.0.slot.1 = alloca i32, align 4
  store i32 1, ptr %toedt.0.slot.1, align 4
  %26 = load i32, ptr %toedt.0.slot.1, align 4
  call void @artsSignalEdt(i64 %3, i32 %26, i64 %5)
  ret void
}

declare i64 @artsDbCreatePtr(ptr, i64, i32)

declare i64 @artsEdtCreateWithGuid(ptr, i64, i32, ptr, i32)

declare void @artsSignalEdt(i64, i32, i64)

define dso_local void @initPerNode(i32 %nodeId, i32 %argc, ptr %argv) {
entry:
  ret void
}

define dso_local void @initPerWorker(i32 %nodeId, i32 %workerId, i32 %argc, ptr %argv) {
entry:
  %0 = icmp eq i32 %nodeId, 0
  %1 = icmp eq i32 %workerId, 0
  %2 = and i1 %0, %1
  br i1 %2, label %then, label %exit

then:                                             ; preds = %entry
  %3 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %edt_2.main_guid.addr = alloca i64, align 8
  store i64 %3, ptr %edt_2.main_guid.addr, align 8
  %4 = load i64, ptr %edt_2.main_guid.addr, align 8
  br label %body

body:                                             ; preds = %then
  %edt_2_paramc = alloca i32, align 4
  store i32 0, ptr %edt_2_paramc, align 4
  %5 = load i32, ptr %edt_2_paramc, align 4
  %edt_2_paramv = alloca i64, i32 %5, align 8
  %6 = alloca i32, align 4
  store i32 0, ptr %6, align 4
  %7 = load i32, ptr %6, align 4
  %8 = call i64 @artsEdtCreateWithGuid(ptr @edt_2.main, i64 %4, i32 %5, ptr %edt_2_paramv, i32 %7)
  br label %exit

exit:                                             ; preds = %body, %entry
  ret void
}

define dso_local i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @artsRT(i32 %argc, ptr %argv)
  ret i32 0
}

declare void @artsRT(i32, ptr)

declare void @artsShutdown()

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { convergent nounwind }
attributes #4 = { nounwind }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"openmp", i32 50}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 16.0.6 (Red Hat 16.0.6-1.el9)"}
!4 = !{!5}
!5 = !{i64 2, i64 -1, i64 -1, i1 true}
!6 = !{!7}
!7 = distinct !{!7, !8, !".omp_outlined..2: argument 0"}
!8 = distinct !{!8, !".omp_outlined..2"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C++ TBAA"}
