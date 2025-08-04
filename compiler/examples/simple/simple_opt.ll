; ModuleID = 'simple_opt.bc'
source_filename = "simple.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

%struct.artsEdtDep_t = type { i64, i32, ptr }

@.str = private unnamed_addr constant [36 x i8] c"EDT 1: The initial number is %d/%d\0A\00", align 1
@.str.1 = private unnamed_addr constant [28 x i8] c"EDT 0: The number is %d/%d\0A\00", align 1
@.str.3 = private unnamed_addr constant [28 x i8] c"EDT 3: The number is %d/%d\0A\00", align 1
@.str.4 = private unnamed_addr constant [37 x i8] c"EDT 2: The final number is %d - %d.\0A\00", align 1

; Function Attrs: nounwind
declare dso_local void @srand(i32 noundef) local_unnamed_addr #0

; Function Attrs: nounwind
declare dso_local i64 @time(ptr noundef) local_unnamed_addr #0

; Function Attrs: nounwind
declare dso_local i32 @rand() local_unnamed_addr #0

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

define internal void @edt_1.task(i32 %paramc, ptr nocapture readonly %paramv, i32 %depc, ptr nocapture readonly %depv) {
entry:
  %0 = load i64, ptr %paramv, align 8
  %1 = trunc i64 %0 to i32
  %edt_1.paramv_1.guid.edt_3 = getelementptr inbounds i64, ptr %paramv, i64 1
  %2 = load i64, ptr %edt_1.paramv_1.guid.edt_3, align 8
  %edt_1.depv_0.guid.load = load i64, ptr %depv, align 8
  %edt_1.depv_0.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i64 0, i32 2
  %edt_1.depv_0.ptr.load = load ptr, ptr %edt_1.depv_0.ptr, align 8
  %3 = load i32, ptr %edt_1.depv_0.ptr.load, align 4, !tbaa !4, !noalias !8
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr %edt_1.depv_0.ptr.load, align 4, !tbaa !4, !noalias !8
  %5 = add nsw i32 %1, 1
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %4, i32 noundef %5) #3, !noalias !8
  tail call void @artsSignalEdt(i64 %2, i32 0, i64 %edt_1.depv_0.guid.load)
  ret void
}

define internal void @edt_0.sync(i32 %paramc, ptr nocapture readonly %paramv, i32 %depc, ptr nocapture readonly %depv) {
entry:
  %0 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %1 = load i64, ptr %paramv, align 8
  %edt_0.depv_0.guid.load = load i64, ptr %depv, align 8
  %edt_0.depv_0.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i64 0, i32 2
  %edt_0.depv_0.ptr.load = load ptr, ptr %edt_0.depv_0.ptr, align 8
  %edt_0.depv_1.guid = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i64 1, i32 0
  %edt_0.depv_1.guid.load = load i64, ptr %edt_0.depv_1.guid, align 8
  %edt_0.depv_1.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i64 1, i32 2
  %edt_0.depv_1.ptr.load = load ptr, ptr %edt_0.depv_1.ptr, align 8
  %2 = load i32, ptr %edt_0.depv_0.ptr.load, align 4, !tbaa !4
  %3 = load i32, ptr %edt_0.depv_1.ptr.load, align 4, !tbaa !4
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %2, i32 noundef %3) #3
  %5 = load i32, ptr %edt_0.depv_1.ptr.load, align 4, !tbaa !4
  %edt_1_paramv1 = alloca [2 x i64], align 8
  %6 = sext i32 %5 to i64
  store i64 %6, ptr %edt_1_paramv1, align 8
  %edt_1.paramv_1.guid.edt_3 = getelementptr inbounds i64, ptr %edt_1_paramv1, i64 1
  store i64 %1, ptr %edt_1.paramv_1.guid.edt_3, align 8
  %7 = call i64 @artsEdtCreateWithGuid(ptr nonnull @edt_1.task, i64 %0, i32 2, ptr nonnull %edt_1_paramv1, i32 1)
  call void @artsSignalEdt(i64 %0, i32 0, i64 %edt_0.depv_0.guid.load)
  call void @artsSignalEdt(i64 %1, i32 1, i64 %edt_0.depv_1.guid.load)
  ret void
}

declare i64 @artsReserveGuidRoute(i32, i32) local_unnamed_addr

define internal void @edt_3.task(i32 %paramc, ptr nocapture readnone %paramv, i32 %depc, ptr nocapture readonly %depv) {
entry:
  %edt_3.depv_0.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i64 0, i32 2
  %edt_3.depv_0.ptr.load = load ptr, ptr %edt_3.depv_0.ptr, align 8
  %edt_3.depv_1.ptr = getelementptr inbounds %struct.artsEdtDep_t, ptr %depv, i64 1, i32 2
  %edt_3.depv_1.ptr.load = load ptr, ptr %edt_3.depv_1.ptr, align 8
  %0 = load i32, ptr %edt_3.depv_0.ptr.load, align 4, !tbaa !4
  %1 = add nsw i32 %0, 1
  store i32 %1, ptr %edt_3.depv_0.ptr.load, align 4, !tbaa !4
  %2 = load i32, ptr %edt_3.depv_1.ptr.load, align 4, !tbaa !4
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %1, i32 noundef %2) #3
  tail call void @artsShutdown()
  ret void
}

define internal void @edt_2.main(i32 %paramc, ptr nocapture readnone %paramv, i32 %depc, ptr nocapture readnone %depv) {
entry:
  %edt_3_paramv2 = alloca [0 x i64], align 8
  %0 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %1 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %db.0.addr = alloca i64, align 8
  %2 = call i64 @artsDbCreatePtr(ptr nonnull %db.0.addr, i64 4, i32 7)
  %db.0.addr.ld = load i64, ptr %db.0.addr, align 8
  %db.0.ptr = inttoptr i64 %db.0.addr.ld to ptr
  %db.1.addr = alloca i64, align 8
  %3 = call i64 @artsDbCreatePtr(ptr nonnull %db.1.addr, i64 4, i32 7)
  %db.1.addr.ld = load i64, ptr %db.1.addr, align 8
  %db.1.ptr = inttoptr i64 %db.1.addr.ld to ptr
  %4 = tail call i64 @time(ptr noundef null) #3
  %5 = trunc i64 %4 to i32
  tail call void @srand(i32 noundef %5) #3
  %6 = tail call i32 @rand() #3
  %7 = srem i32 %6, 100
  %8 = add nsw i32 %7, 1
  store i32 %8, ptr %db.0.ptr, align 4, !tbaa !4
  %9 = tail call i32 @rand() #3
  %10 = srem i32 %9, 10
  %11 = add nsw i32 %10, 1
  store i32 %11, ptr %db.1.ptr, align 4, !tbaa !4
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %8, i32 noundef %11) #3
  %edt_0_paramv = alloca i64, align 8
  store i64 %0, ptr %edt_0_paramv, align 8
  %13 = call i64 @artsEdtCreateWithGuid(ptr nonnull @edt_0.sync, i64 %1, i32 1, ptr nonnull %edt_0_paramv, i32 2)
  %14 = call i64 @artsEdtCreateWithGuid(ptr nonnull @edt_3.task, i64 %0, i32 0, ptr nonnull %edt_3_paramv2, i32 2)
  call void @artsSignalEdt(i64 %1, i32 0, i64 %2)
  call void @artsSignalEdt(i64 %1, i32 1, i64 %3)
  ret void
}

declare i64 @artsDbCreatePtr(ptr, i64, i32) local_unnamed_addr

declare i64 @artsEdtCreateWithGuid(ptr, i64, i32, ptr, i32) local_unnamed_addr

declare void @artsSignalEdt(i64, i32, i64) local_unnamed_addr

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local void @initPerNode(i32 %nodeId, i32 %argc, ptr nocapture readnone %argv) local_unnamed_addr #2 {
entry:
  ret void
}

define dso_local void @initPerWorker(i32 %nodeId, i32 %workerId, i32 %argc, ptr nocapture readnone %argv) local_unnamed_addr {
entry:
  %edt_2_paramv1 = alloca [0 x i64], align 8
  %0 = or i32 %workerId, %nodeId
  %1 = icmp eq i32 %0, 0
  br i1 %1, label %then, label %exit

then:                                             ; preds = %entry
  %2 = tail call i64 @artsReserveGuidRoute(i32 1, i32 0)
  %3 = call i64 @artsEdtCreateWithGuid(ptr nonnull @edt_2.main, i64 %2, i32 0, ptr nonnull %edt_2_paramv1, i32 0)
  br label %exit

exit:                                             ; preds = %then, %entry
  ret void
}

define dso_local noundef i32 @main(i32 %argc, ptr %argv) local_unnamed_addr {
entry:
  tail call void @artsRT(i32 %argc, ptr %argv)
  ret i32 0
}

declare void @artsRT(i32, ptr) local_unnamed_addr

declare void @artsShutdown() local_unnamed_addr

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"openmp", i32 50}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 16.0.6 (Red Hat 16.0.6-1.el9)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9}
!9 = distinct !{!9, !10, !".omp_outlined..2: argument 0"}
!10 = distinct !{!10, !".omp_outlined..2"}
