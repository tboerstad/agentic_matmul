	.att_syntax
	.file	"asm_driver.mojo"
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:
.Lmain$local:
	.type	.Lmain$local,@function
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$200, %rsp
	.cfi_def_cfa_offset 256
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movl	%edi, %ebp
	callq	KGEN_CompilerRT_AsyncRT_GetCurrentRuntime@PLT
	testq	%rax, %rax
	jne	.LBB0_2
	leaq	static_string_a61c3395ab9379d9(%rip), %rdi
	movq	"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_0"@GOTPCREL(%rip), %rdx
	movq	"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_1"@GOTPCREL(%rip), %rcx
	movl	$7, %esi
	callq	KGEN_CompilerRT_GetOrCreateGlobal@PLT
.LBB0_2:
	movl	%ebp, %edi
	movq	%rbx, %rsi
	callq	KGEN_CompilerRT_SetArgV@PLT
	callq	KGEN_CompilerRT_PrintStackTraceOnFault@PLT
	movl	$8, %edi
	movl	$1572864, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %rbx
	movl	$196608, %ecx
	movq	$-196608, %r14
	xorl	%edx, %edx
	jmp	.LBB0_3
	.p2align	4
.LBB0_5:
	movq	$0, (%rbx,%rdx,8)
	incq	%rdx
	incq	%r14
	je	.LBB0_6
.LBB0_3:
	cmpq	%rcx, %rdx
	jl	.LBB0_5
	xorl	%eax, %eax
	testq	%rcx, %rcx
	sete	%al
	leaq	(%rax,%rcx,2), %rax
	movq	%rbx, %rdi
	movq	%rdx, %rsi
	movq	%rcx, %rdx
	movq	%rax, %rcx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::builtin::simd::SIMD,dtype=f64,size=1\">>, scalar<f64>]"@PLT
	movq	%rax, %rbx
	jmp	.LBB0_5
.LBB0_6:
	movl	$8, %edi
	movl	$180355072, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r15
	movl	$22544384, %ecx
	movq	$-22544384, %r14
	xorl	%edx, %edx
	jmp	.LBB0_7
	.p2align	4
.LBB0_9:
	movq	$0, (%r15,%rdx,8)
	incq	%rdx
	incq	%r14
	je	.LBB0_10
.LBB0_7:
	cmpq	%rcx, %rdx
	jl	.LBB0_9
	xorl	%eax, %eax
	testq	%rcx, %rcx
	sete	%al
	leaq	(%rax,%rcx,2), %rax
	movq	%r15, %rdi
	movq	%rdx, %rsi
	movq	%rcx, %rdx
	movq	%rax, %rcx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::builtin::simd::SIMD,dtype=f64,size=1\">>, scalar<f64>]"@PLT
	movq	%rax, %r15
	jmp	.LBB0_9
.LBB0_10:
	movl	$8, %edi
	movl	$8454144, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r12
	movl	$1056768, %ecx
	movq	$-1056768, %r14
	xorl	%edx, %edx
	jmp	.LBB0_11
	.p2align	4
.LBB0_13:
	movq	$0, (%r12,%rdx,8)
	incq	%rdx
	incq	%r14
	je	.LBB0_14
.LBB0_11:
	cmpq	%rcx, %rdx
	jl	.LBB0_13
	xorl	%eax, %eax
	testq	%rcx, %rcx
	sete	%al
	leaq	(%rax,%rcx,2), %rax
	movq	%r12, %rdi
	movq	%rdx, %rsi
	movq	%rcx, %rdx
	movq	%rax, %rcx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::builtin::simd::SIMD,dtype=f64,size=1\">>, scalar<f64>]"@PLT
	movq	%rax, %r12
	jmp	.LBB0_13
.LBB0_14:
	movl	$95, %eax
	xorl	%edx, %edx
	movq	%rbx, %rcx
	.p2align	4
.LBB0_15:
	imulq	$88064, %rdx, %rdx
	addq	%r12, %rdx
	movl	$11007, %esi
	movq	%r15, %rdi
	xorl	%r8d, %r8d
	.p2align	4
.LBB0_16:
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%r9d, %r9d
	movq	%rdi, %r10
	.p2align	4
.LBB0_17:
	vmovsd	(%rcx,%r9,8), %xmm1
	vfmadd231sd	(%r10), %xmm1, %xmm0
	addq	$88064, %r10
	incq	%r9
	cmpq	$2048, %r9
	jne	.LBB0_17
	vmovsd	%xmm0, (%rdx,%r8,8)
	movl	$11008, %r8d
	subq	%rsi, %r8
	addq	$8, %rdi
	subq	$1, %rsi
	jae	.LBB0_16
	movl	$96, %edx
	subq	%rax, %rdx
	addq	$16384, %rcx
	subq	$1, %rax
	jae	.LBB0_15
	xorl	%r14d, %r14d
	movl	$8454144, %edx
	movq	%r12, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r12, 160(%rsp)
	movq	%r12, 136(%rsp)
	movq	%r15, 168(%rsp)
	.p2align	4
.LBB0_21:
	leaq	32(%r14), %rcx
	movq	%r14, %rax
	orq	$1, %rax
	movq	%rax, 48(%rsp)
	movq	%r15, %rsi
	xorl	%edx, %edx
	.p2align	4
.LBB0_22:
	leaq	32(%rdx), %r8
	movq	%rdx, %r9
	orq	$1, %r9
	movq	%rsi, 128(%rsp)
	movq	136(%rsp), %r11
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_23:
	leaq	32(%r15), %rax
	movq	%rax, 56(%rsp)
	movq	%r11, %r13
	movq	48(%rsp), %rax
	movq	%r14, %r12
	.p2align	4
.LBB0_24:
	movq	%r14, %rbp
	movq	%rax, %r14
	shlq	$14, %rbp
	addq	%rbx, %rbp
	movq	%rsi, %r10
	movq	%r9, %rdi
	movq	%rdx, %rax
	.p2align	4
.LBB0_25:
	vmovsd	(%rbp,%rax,8), %xmm0
	movq	%rdi, %rax
	xorl	%edi, %edi
	.p2align	4
.LBB0_26:
	vmovsd	(%r10,%rdi,8), %xmm1
	vfmadd213sd	(%r13,%rdi,8), %xmm0, %xmm1
	vmovsd	%xmm1, (%r13,%rdi,8)
	incq	%rdi
	cmpq	$32, %rdi
	jne	.LBB0_26
	xorl	%edi, %edi
	cmpq	%r8, %rax
	setne	%dil
	addq	%rax, %rdi
	addq	$88064, %r10
	cmpq	%r8, %rax
	jne	.LBB0_25
	xorl	%eax, %eax
	cmpq	%rcx, %r14
	setne	%al
	addq	%r14, %rax
	addq	$88064, %r13
	cmpq	%rcx, %r14
	jne	.LBB0_24
	addq	$256, %r11
	addq	$256, %rsi
	cmpq	$10976, %r15
	movq	56(%rsp), %r15
	movq	%r12, %r14
	jb	.LBB0_23
	movq	128(%rsp), %rsi
	addq	$2818048, %rsi
	cmpq	$2016, %rdx
	movq	%r8, %rdx
	jb	.LBB0_22
	addq	$2818048, 136(%rsp)
	cmpq	$64, %r14
	movq	%rcx, %r14
	movq	168(%rsp), %r15
	jb	.LBB0_21
	xorl	%r13d, %r13d
	movl	$8454144, %edx
	movq	160(%rsp), %r14
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r14, 128(%rsp)
	.p2align	4
.LBB0_33:
	leaq	32(%r13), %rcx
	movq	%r13, %rax
	orq	$1, %rax
	movq	%rax, 56(%rsp)
	movq	%r15, %rsi
	xorl	%r14d, %r14d
	.p2align	4
.LBB0_34:
	leaq	32(%r14), %r8
	movq	%r14, %r9
	orq	$1, %r9
	movq	128(%rsp), %r10
	movq	%rsi, 48(%rsp)
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_35:
	movq	%r10, %r12
	movq	56(%rsp), %rax
	movq	%r13, %rdx
	.p2align	4
.LBB0_36:
	movq	%r13, %rbp
	movq	%rax, %r13
	shlq	$14, %rbp
	addq	%rbx, %rbp
	movq	%rsi, %r11
	movq	%r9, %rdi
	movq	%r14, %rax
	.p2align	4
.LBB0_37:
	vbroadcastsd	(%rbp,%rax,8), %zmm0
	movq	%rdi, %rax
	movq	$-8, %rdi
	.p2align	4
.LBB0_38:
	vmovupd	64(%r11,%rdi,8), %zmm1
	vfmadd213pd	64(%r12,%rdi,8), %zmm0, %zmm1
	vmovupd	%zmm1, 64(%r12,%rdi,8)
	addq	$8, %rdi
	cmpq	$24, %rdi
	jb	.LBB0_38
	xorl	%edi, %edi
	cmpq	%r8, %rax
	setne	%dil
	addq	%rax, %rdi
	addq	$88064, %r11
	cmpq	%r8, %rax
	jne	.LBB0_37
	xorl	%eax, %eax
	cmpq	%rcx, %r13
	setne	%al
	addq	%r13, %rax
	addq	$88064, %r12
	cmpq	%rcx, %r13
	jne	.LBB0_36
	addq	$256, %rsi
	addq	$256, %r10
	cmpq	$10976, %r15
	leaq	32(%r15), %r15
	movq	%rdx, %r13
	jb	.LBB0_35
	movq	48(%rsp), %rsi
	addq	$2818048, %rsi
	cmpq	$2016, %r14
	movq	%r8, %r14
	jb	.LBB0_34
	addq	$2818048, 128(%rsp)
	cmpq	$64, %r13
	movq	%rcx, %r13
	movq	168(%rsp), %r15
	jb	.LBB0_33
	movq	$96, 64(%rsp)
	movq	$11008, 72(%rsp)
	movq	$2048, 80(%rsp)
	movq	160(%rsp), %r14
	movq	%r14, 88(%rsp)
	movq	%r15, 96(%rsp)
	movq	%rbx, 104(%rsp)
	movl	$8454144, %edx
	movq	%r14, %rdi
	xorl	%esi, %esi
	vzeroupper
	callq	memset@PLT
	callq	KGEN_CompilerRT_NumPhysicalCores@PLT
	movq	%rax, %r12
	cmpq	$1, %rax
	movq	%rax, %rcx
	adcq	$0, %rcx
	movq	%rcx, %rax
	shrq	$32, %rax
	je	.LBB0_45
	movl	$3, %eax
	xorl	%edx, %edx
	idivq	%rcx
	movq	%rax, %r13
	jmp	.LBB0_47
.LBB0_45:
	movl	$3, %eax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %r13d
.LBB0_47:
	testq	%r12, %r12
	sets	%al
	testq	%rdx, %rdx
	setne	%cl
	andb	%al, %cl
	movzbl	%cl, %eax
	subq	%rax, %r13
	xorl	%ebp, %ebp
	testb	%al, %al
	cmovneq	%r12, %rbp
	addq	%rdx, %rbp
	testq	%r12, %r12
	cmoveq	%r12, %r13
	movq	%r13, 112(%rsp)
	cmoveq	%r12, %rbp
	movq	%rbp, 120(%rsp)
	movq	%r12, 128(%rsp)
	movq	%r13, 136(%rsp)
	movq	%rbp, 176(%rsp)
	jle	.LBB0_91
	cmpq	$1, %r12
	jne	.LBB0_72
	movl	$0, 8(%rsp)
	vstmxcsr	8(%rsp)
	movl	8(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB0_51
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
.LBB0_51:
	movl	%ecx, 56(%rsp)
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_52
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r13
	movq	%rdx, %r15
	leaq	(%rdx,%rdx,2), %rax
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rcx
	movq	%rcx, (%r13,%rax,8)
	movq	$5, 8(%r13,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r13,%rax,8)
	incq	%r15
	jmp	.LBB0_54
.LBB0_72:
	callq	KGEN_CompilerRT_AsyncRT_ParallelismLevel@PLT
	movl	%eax, 156(%rsp)
	movslq	%eax, %r15
	testl	%r15d, %r15d
	movl	$1, %ecx
	cmovneq	%r15, %rcx
	movq	%r12, %rax
	orq	%rcx, %rax
	shrq	$32, %rax
	je	.LBB0_73
	movq	%r12, %rax
	cqto
	idivq	%rcx
	movq	%rax, %r12
	jmp	.LBB0_75
.LBB0_52:
	xorl	%r13d, %r13d
	xorl	%r15d, %r15d
.LBB0_54:
	leaq	static_string_44fd141e40b306d5(%rip), %rdi
	movl	$2, %esi
	movq	%r13, %rdx
	movq	%r15, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r12
	movq	%rcx, %rbp
	xorl	%r14d, %r14d
	testq	%r15, %r15
	cmovgq	%r15, %r14
	movabsq	$4611686018427387904, %rdx
	jle	.LBB0_60
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %r15
	jmp	.LBB0_56
	.p2align	4
.LBB0_59:
	movq	%r15, %rax
	addq	$-1, %rax
	jae	.LBB0_60
.LBB0_56:
	movq	%r14, %rcx
	subq	%r15, %rcx
	movq	%rax, %r15
	leaq	(%rcx,%rcx,2), %rax
	testq	%rdx, 16(%r13,%rax,8)
	je	.LBB0_59
	leaq	(,%rax,8), %rax
	addq	%r13, %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_59
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movabsq	$4611686018427387904, %rdx
	jmp	.LBB0_59
.LBB0_60:
	movq	%r13, %rdi
	movq	%rdx, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%r14, %rbp
	je	.LBB0_63
	lock		decq	-8(%r12)
	jne	.LBB0_63
	addq	$-8, %r12
	#MEMBARRIER
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_63:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_64
	leaq	static_string_2b3f504061b33816(%rip), %rdi
	movl	$4, %esi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, 48(%rsp)
	jmp	.LBB0_66
.LBB0_73:
	movl	%r12d, %eax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %r12d
.LBB0_75:
	testl	%r15d, %r15d
	sets	%al
	movq	%rdx, 184(%rsp)
	testq	%rdx, %rdx
	setne	%bpl
	xorl	%r14d, %r14d
	andb	%al, %bpl
	movl	$0, %eax
	cmovneq	%r15, %rax
	movq	%rax, 192(%rsp)
	movq	$0, 144(%rsp)
	leaq	144(%rsp), %rdi
	callq	KGEN_CompilerRT_AsyncRT_InitializeChain@PLT
	movq	$1, 8(%rsp)
	movq	144(%rsp), %rax
	movq	%rax, 16(%rsp)
	movl	$8, %edi
	movl	$128, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, 24(%rsp)
	movq	$0, 32(%rsp)
	movq	$16, 40(%rsp)
	testl	%r15d, %r15d
	je	.LBB0_78
	movzbl	%bpl, %eax
	subq	%rax, %r12
	testq	%r12, %r12
	jle	.LBB0_78
	movq	%r15, %rax
	sarq	$63, %rax
	andnq	%r15, %rax, %r13
	cmpq	$1, %r13
	movq	%r13, %rax
	adcq	$-1, %rax
	movq	%rax, 48(%rsp)
	xorl	%r14d, %r14d
	testl	%r15d, %r15d
	jg	.LBB0_98
.LBB0_78:
	cmpl	$0, 156(%rsp)
	movq	192(%rsp), %r15
	je	.LBB0_84
	addq	184(%rsp), %r15
	testq	%r15, %r15
	jle	.LBB0_84
	leaq	8(%rsp), %r12
	movq	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"@GOTPCREL(%rip), %r13
	jmp	.LBB0_81
	.p2align	4
.LBB0_83:
	decq	%r15
	movq	%rbp, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	testq	%r15, %r15
	je	.LBB0_84
.LBB0_81:
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %rbp
	movl	$0, (%rax)
	movq	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%rbp)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%rbp)
	movq	$4, 56(%rbp)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%rbp)
	movq	$0, 72(%rbp)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%rbp)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%rbp)
	movq	$2, 96(%rbp)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%rbp)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%rbp)
	movq	%r14, 120(%rbp)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%rbp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rbp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rbp)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%rbp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rbp)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%rbp)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%rbp)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%rbp)
	movq	%r12, 24(%rbp)
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	$-1, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_83
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_83
	.p2align	4
.LBB0_97:
	movq	56(%rsp), %r12
	decq	%r12
	je	.LBB0_78
.LBB0_98:
	movq	%r12, 56(%rsp)
	movq	48(%rsp), %rax
	movq	%r13, %rbp
	jmp	.LBB0_99
	.p2align	4
.LBB0_101:
	movq	%r12, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB0_97
.LBB0_99:
	movq	%r13, %r15
	subq	%rbp, %r15
	movq	%rax, %rbp
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r12
	movl	$0, (%rax)
	movq	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%r12)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%r12)
	movq	$4, 56(%r12)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%r12)
	movq	$0, 72(%r12)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%r12)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%r12)
	movq	$2, 96(%r12)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%r12)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%r12)
	movq	%r14, 120(%r12)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%r12)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%r12)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%r12)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%r12)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%r12)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%r12)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%r12)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%r12)
	leaq	8(%rsp), %rax
	movq	%rax, 24(%r12)
	movq	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"@GOTPCREL(%rip), %rdi
	movq	%r12, %rsi
	movq	%r15, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_101
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_101
.LBB0_84:
	leaq	16(%rsp), %r12
	lock		decq	8(%rsp)
	jne	.LBB0_86
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Complete@PLT
.LBB0_86:
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Wait@PLT
	movq	24(%rsp), %r15
	movq	32(%rsp), %rax
	xorl	%r14d, %r14d
	testq	%rax, %rax
	cmovgq	%rax, %r14
	movq	176(%rsp), %rbp
	jle	.LBB0_90
	xorl	%r13d, %r13d
	.p2align	4
.LBB0_88:
	movq	(%r15,%r13,8), %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	incq	%r13
	cmpq	%r13, %r14
	jne	.LBB0_88
	movq	24(%rsp), %r15
.LBB0_90:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_DestroyChain@PLT
	movq	168(%rsp), %r15
	movq	160(%rsp), %r14
	movq	128(%rsp), %r12
	movq	136(%rsp), %r13
	jmp	.LBB0_91
.LBB0_64:
	movq	$0, 48(%rsp)
.LBB0_66:
	movq	120(%rsp), %rax
	xorl	%ebp, %ebp
	testq	%rax, %rax
	setg	%bpl
	addq	112(%rsp), %rbp
	testq	%rbp, %rbp
	jle	.LBB0_69
	movq	%rax, %r15
	sarq	$63, %r15
	andq	%rax, %r15
	leaq	104(%rsp), %r12
	leaq	72(%rsp), %r14
	leaq	80(%rsp), %r13
	.p2align	4
.LBB0_68:
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	movq	%r15, %rdi
	leaq	96(%rsp), %rsi
	leaq	88(%rsp), %rdx
	movq	%r12, %rcx
	movq	%r14, %r8
	movq	%r13, %r9
	callq	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	incq	%r15
	decq	%rbp
	jne	.LBB0_68
.LBB0_69:
	movq	48(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_71
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
.LBB0_71:
	movl	56(%rsp), %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
	movq	168(%rsp), %r15
	movq	160(%rsp), %r14
	movq	128(%rsp), %r12
	movq	136(%rsp), %r13
	movq	176(%rsp), %rbp
.LBB0_91:
	movq	$96, 64(%rsp)
	movq	$11008, 72(%rsp)
	movq	$2048, 80(%rsp)
	movq	%r14, 88(%rsp)
	movq	%r15, 96(%rsp)
	movq	%rbx, 104(%rsp)
	movl	$8454144, %edx
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r13, 112(%rsp)
	movq	%rbp, 120(%rsp)
	testq	%r12, %r12
	jle	.LBB0_140
	cmpq	$1, %r12
	jne	.LBB0_121
	movl	$0, 8(%rsp)
	vstmxcsr	8(%rsp)
	movl	8(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB0_95
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
.LBB0_95:
	movl	%ecx, 56(%rsp)
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_96
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r13
	movq	%rdx, %r15
	leaq	(%rdx,%rdx,2), %rax
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rcx
	movq	%rcx, (%r13,%rax,8)
	movq	$5, 8(%r13,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r13,%rax,8)
	incq	%r15
	jmp	.LBB0_103
.LBB0_121:
	callq	KGEN_CompilerRT_AsyncRT_ParallelismLevel@PLT
	movl	%eax, 156(%rsp)
	movslq	%eax, %r15
	testl	%r15d, %r15d
	movl	$1, %ecx
	cmovneq	%r15, %rcx
	movq	%r12, %rax
	orq	%rcx, %rax
	shrq	$32, %rax
	je	.LBB0_122
	movq	%r12, %rax
	cqto
	idivq	%rcx
	movq	%rax, %r12
	jmp	.LBB0_124
.LBB0_96:
	xorl	%r13d, %r13d
	xorl	%r15d, %r15d
.LBB0_103:
	leaq	static_string_44fd141e40b306d5(%rip), %rdi
	movl	$2, %esi
	movq	%r13, %rdx
	movq	%r15, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r12
	movq	%rcx, %rbp
	xorl	%r14d, %r14d
	testq	%r15, %r15
	cmovgq	%r15, %r14
	movabsq	$4611686018427387904, %rdx
	jle	.LBB0_109
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %r15
	jmp	.LBB0_105
	.p2align	4
.LBB0_108:
	movq	%r15, %rax
	addq	$-1, %rax
	jae	.LBB0_109
.LBB0_105:
	movq	%r14, %rcx
	subq	%r15, %rcx
	movq	%rax, %r15
	leaq	(%rcx,%rcx,2), %rax
	testq	%rdx, 16(%r13,%rax,8)
	je	.LBB0_108
	leaq	(,%rax,8), %rax
	addq	%r13, %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_108
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movabsq	$4611686018427387904, %rdx
	jmp	.LBB0_108
.LBB0_109:
	movq	%r13, %rdi
	movq	%rdx, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%r14, %rbp
	je	.LBB0_112
	lock		decq	-8(%r12)
	jne	.LBB0_112
	addq	$-8, %r12
	#MEMBARRIER
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_112:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_113
	leaq	static_string_2b3f504061b33816(%rip), %rdi
	movl	$4, %esi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, 48(%rsp)
	jmp	.LBB0_115
.LBB0_122:
	movl	%r12d, %eax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %r12d
.LBB0_124:
	testl	%r15d, %r15d
	sets	%al
	movq	%rdx, 184(%rsp)
	testq	%rdx, %rdx
	setne	%bpl
	xorl	%r14d, %r14d
	andb	%al, %bpl
	movl	$0, %eax
	cmovneq	%r15, %rax
	movq	%rax, 192(%rsp)
	movq	$0, 144(%rsp)
	leaq	144(%rsp), %rdi
	callq	KGEN_CompilerRT_AsyncRT_InitializeChain@PLT
	movq	$1, 8(%rsp)
	movq	144(%rsp), %rax
	movq	%rax, 16(%rsp)
	movl	$8, %edi
	movl	$128, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, 24(%rsp)
	movq	$0, 32(%rsp)
	movq	$16, 40(%rsp)
	testl	%r15d, %r15d
	je	.LBB0_127
	movzbl	%bpl, %eax
	subq	%rax, %r12
	testq	%r12, %r12
	jle	.LBB0_127
	movq	%r15, %rax
	sarq	$63, %rax
	andnq	%r15, %rax, %r13
	cmpq	$1, %r13
	movq	%r13, %rax
	adcq	$-1, %rax
	movq	%rax, 48(%rsp)
	xorl	%r14d, %r14d
	testl	%r15d, %r15d
	jg	.LBB0_147
.LBB0_127:
	cmpl	$0, 156(%rsp)
	movq	192(%rsp), %r15
	je	.LBB0_133
	addq	184(%rsp), %r15
	testq	%r15, %r15
	jle	.LBB0_133
	leaq	8(%rsp), %r12
	movq	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"@GOTPCREL(%rip), %r13
	jmp	.LBB0_130
	.p2align	4
.LBB0_132:
	decq	%r15
	movq	%rbp, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	testq	%r15, %r15
	je	.LBB0_133
.LBB0_130:
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %rbp
	movl	$0, (%rax)
	movq	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%rbp)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%rbp)
	movq	$4, 56(%rbp)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%rbp)
	movq	$0, 72(%rbp)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%rbp)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%rbp)
	movq	$2, 96(%rbp)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%rbp)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%rbp)
	movq	%r14, 120(%rbp)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%rbp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rbp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rbp)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%rbp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rbp)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%rbp)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%rbp)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%rbp)
	movq	%r12, 24(%rbp)
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	$-1, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_132
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_132
	.p2align	4
.LBB0_146:
	movq	56(%rsp), %r12
	decq	%r12
	je	.LBB0_127
.LBB0_147:
	movq	%r12, 56(%rsp)
	movq	48(%rsp), %rax
	movq	%r13, %rbp
	jmp	.LBB0_148
	.p2align	4
.LBB0_150:
	movq	%r12, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB0_146
.LBB0_148:
	movq	%r13, %r15
	subq	%rbp, %r15
	movq	%rax, %rbp
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r12
	movl	$0, (%rax)
	movq	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%r12)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%r12)
	movq	$4, 56(%r12)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%r12)
	movq	$0, 72(%r12)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%r12)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%r12)
	movq	$2, 96(%r12)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%r12)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%r12)
	movq	%r14, 120(%r12)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%r12)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%r12)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%r12)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%r12)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%r12)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%r12)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%r12)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%r12)
	leaq	8(%rsp), %rax
	movq	%rax, 24(%r12)
	movq	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"@GOTPCREL(%rip), %rdi
	movq	%r12, %rsi
	movq	%r15, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_150
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_150
.LBB0_133:
	leaq	16(%rsp), %r12
	lock		decq	8(%rsp)
	jne	.LBB0_135
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Complete@PLT
.LBB0_135:
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Wait@PLT
	movq	24(%rsp), %r15
	movq	32(%rsp), %rax
	xorl	%r14d, %r14d
	testq	%rax, %rax
	cmovgq	%rax, %r14
	movq	176(%rsp), %rbp
	jle	.LBB0_139
	xorl	%r13d, %r13d
	.p2align	4
.LBB0_137:
	movq	(%r15,%r13,8), %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	incq	%r13
	cmpq	%r13, %r14
	jne	.LBB0_137
	movq	24(%rsp), %r15
.LBB0_139:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_DestroyChain@PLT
	movq	168(%rsp), %r15
	movq	160(%rsp), %r14
	movq	128(%rsp), %r12
	movq	136(%rsp), %r13
	jmp	.LBB0_140
.LBB0_113:
	movq	$0, 48(%rsp)
.LBB0_115:
	movq	120(%rsp), %rax
	xorl	%ebp, %ebp
	testq	%rax, %rax
	setg	%bpl
	addq	112(%rsp), %rbp
	testq	%rbp, %rbp
	jle	.LBB0_118
	movq	%rax, %r15
	sarq	$63, %r15
	andq	%rax, %r15
	leaq	104(%rsp), %r12
	leaq	72(%rsp), %r14
	leaq	80(%rsp), %r13
	.p2align	4
.LBB0_117:
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	movq	%r15, %rdi
	leaq	96(%rsp), %rsi
	leaq	88(%rsp), %rdx
	movq	%r12, %rcx
	movq	%r14, %r8
	movq	%r13, %r9
	callq	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	incq	%r15
	decq	%rbp
	jne	.LBB0_117
.LBB0_118:
	movq	48(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_120
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
.LBB0_120:
	movl	56(%rsp), %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
	movq	168(%rsp), %r15
	movq	160(%rsp), %r14
	movq	128(%rsp), %r12
	movq	136(%rsp), %r13
	movq	176(%rsp), %rbp
.LBB0_140:
	movq	$96, 64(%rsp)
	movq	$11008, 72(%rsp)
	movq	$2048, 80(%rsp)
	movq	%r14, 88(%rsp)
	movq	%r15, 96(%rsp)
	movq	%rbx, 104(%rsp)
	movl	$8454144, %edx
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r13, 112(%rsp)
	movq	%rbp, 120(%rsp)
	testq	%r12, %r12
	jle	.LBB0_189
	cmpq	$1, %r12
	jne	.LBB0_170
	movl	$0, 8(%rsp)
	vstmxcsr	8(%rsp)
	movl	8(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB0_144
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
.LBB0_144:
	movl	%ecx, 56(%rsp)
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_145
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r13
	movq	%rdx, %r15
	leaq	(%rdx,%rdx,2), %rax
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rcx
	movq	%rcx, (%r13,%rax,8)
	movq	$5, 8(%r13,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r13,%rax,8)
	incq	%r15
	jmp	.LBB0_152
.LBB0_170:
	callq	KGEN_CompilerRT_AsyncRT_ParallelismLevel@PLT
	movl	%eax, 156(%rsp)
	movslq	%eax, %r15
	testl	%r15d, %r15d
	movl	$1, %ecx
	cmovneq	%r15, %rcx
	movq	%r12, %rax
	orq	%rcx, %rax
	shrq	$32, %rax
	je	.LBB0_171
	movq	%r12, %rax
	cqto
	idivq	%rcx
	movq	%rax, %r12
	jmp	.LBB0_173
.LBB0_145:
	xorl	%r13d, %r13d
	xorl	%r15d, %r15d
.LBB0_152:
	leaq	static_string_44fd141e40b306d5(%rip), %rdi
	movl	$2, %esi
	movq	%r13, %rdx
	movq	%r15, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r12
	movq	%rcx, %rbp
	xorl	%r14d, %r14d
	testq	%r15, %r15
	cmovgq	%r15, %r14
	movabsq	$4611686018427387904, %rdx
	jle	.LBB0_158
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %r15
	jmp	.LBB0_154
	.p2align	4
.LBB0_157:
	movq	%r15, %rax
	addq	$-1, %rax
	jae	.LBB0_158
.LBB0_154:
	movq	%r14, %rcx
	subq	%r15, %rcx
	movq	%rax, %r15
	leaq	(%rcx,%rcx,2), %rax
	testq	%rdx, 16(%r13,%rax,8)
	je	.LBB0_157
	leaq	(,%rax,8), %rax
	addq	%r13, %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_157
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movabsq	$4611686018427387904, %rdx
	jmp	.LBB0_157
.LBB0_158:
	movq	%r13, %rdi
	movq	%rdx, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%r14, %rbp
	je	.LBB0_161
	lock		decq	-8(%r12)
	jne	.LBB0_161
	addq	$-8, %r12
	#MEMBARRIER
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_161:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_162
	leaq	static_string_2b3f504061b33816(%rip), %rdi
	movl	$4, %esi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, 48(%rsp)
	jmp	.LBB0_164
.LBB0_171:
	movl	%r12d, %eax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %r12d
.LBB0_173:
	testl	%r15d, %r15d
	sets	%al
	movq	%rdx, 184(%rsp)
	testq	%rdx, %rdx
	setne	%bpl
	xorl	%r14d, %r14d
	andb	%al, %bpl
	movl	$0, %eax
	cmovneq	%r15, %rax
	movq	%rax, 192(%rsp)
	movq	$0, 144(%rsp)
	leaq	144(%rsp), %rdi
	callq	KGEN_CompilerRT_AsyncRT_InitializeChain@PLT
	movq	$1, 8(%rsp)
	movq	144(%rsp), %rax
	movq	%rax, 16(%rsp)
	movl	$8, %edi
	movl	$128, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, 24(%rsp)
	movq	$0, 32(%rsp)
	movq	$16, 40(%rsp)
	testl	%r15d, %r15d
	je	.LBB0_176
	movzbl	%bpl, %eax
	subq	%rax, %r12
	testq	%r12, %r12
	jle	.LBB0_176
	movq	%r15, %rax
	sarq	$63, %rax
	andnq	%r15, %rax, %r13
	cmpq	$1, %r13
	movq	%r13, %rax
	adcq	$-1, %rax
	movq	%rax, 48(%rsp)
	xorl	%r14d, %r14d
	testl	%r15d, %r15d
	jg	.LBB0_196
.LBB0_176:
	cmpl	$0, 156(%rsp)
	movq	192(%rsp), %r15
	je	.LBB0_182
	addq	184(%rsp), %r15
	testq	%r15, %r15
	jle	.LBB0_182
	leaq	8(%rsp), %r12
	movq	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"@GOTPCREL(%rip), %r13
	jmp	.LBB0_179
	.p2align	4
.LBB0_181:
	decq	%r15
	movq	%rbp, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	testq	%r15, %r15
	je	.LBB0_182
.LBB0_179:
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %rbp
	movl	$0, (%rax)
	movq	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%rbp)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%rbp)
	movq	$4, 56(%rbp)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%rbp)
	movq	$0, 72(%rbp)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%rbp)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%rbp)
	movq	$2, 96(%rbp)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%rbp)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%rbp)
	movq	%r14, 120(%rbp)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%rbp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rbp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rbp)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%rbp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rbp)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%rbp)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%rbp)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%rbp)
	movq	%r12, 24(%rbp)
	movq	%r13, %rdi
	movq	%rbp, %rsi
	movq	$-1, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_181
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_181
	.p2align	4
.LBB0_195:
	movq	56(%rsp), %r12
	decq	%r12
	je	.LBB0_176
.LBB0_196:
	movq	%r12, 56(%rsp)
	movq	48(%rsp), %rax
	movq	%r13, %rbp
	jmp	.LBB0_197
	.p2align	4
.LBB0_199:
	movq	%r12, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB0_195
.LBB0_197:
	movq	%r13, %r15
	subq	%rbp, %r15
	movq	%rax, %rbp
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r12
	movl	$0, (%rax)
	movq	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%r12)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%r12)
	movq	$4, 56(%r12)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%r12)
	movq	$0, 72(%r12)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%r12)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%r12)
	movq	$2, 96(%r12)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%r12)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%r12)
	movq	%r14, 120(%r12)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%r12)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%r12)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%r12)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%r12)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%r12)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%r12)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%r12)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%r12)
	leaq	8(%rsp), %rax
	movq	%rax, 24(%r12)
	movq	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"@GOTPCREL(%rip), %rdi
	movq	%r12, %rsi
	movq	%r15, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_199
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_199
.LBB0_182:
	leaq	16(%rsp), %r12
	lock		decq	8(%rsp)
	jne	.LBB0_184
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Complete@PLT
.LBB0_184:
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Wait@PLT
	movq	24(%rsp), %r15
	movq	32(%rsp), %rax
	xorl	%r14d, %r14d
	testq	%rax, %rax
	cmovgq	%rax, %r14
	movq	176(%rsp), %rbp
	jle	.LBB0_188
	xorl	%r13d, %r13d
	.p2align	4
.LBB0_186:
	movq	(%r15,%r13,8), %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	incq	%r13
	cmpq	%r13, %r14
	jne	.LBB0_186
	movq	24(%rsp), %r15
.LBB0_188:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_DestroyChain@PLT
	movq	168(%rsp), %r15
	movq	160(%rsp), %r14
	movq	128(%rsp), %r12
	movq	136(%rsp), %r13
	jmp	.LBB0_189
.LBB0_162:
	movq	$0, 48(%rsp)
.LBB0_164:
	movq	120(%rsp), %rax
	xorl	%ebp, %ebp
	testq	%rax, %rax
	setg	%bpl
	addq	112(%rsp), %rbp
	testq	%rbp, %rbp
	jle	.LBB0_167
	movq	%rax, %r15
	sarq	$63, %r15
	andq	%rax, %r15
	leaq	104(%rsp), %r12
	leaq	72(%rsp), %r14
	leaq	80(%rsp), %r13
	.p2align	4
.LBB0_166:
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	movq	%r15, %rdi
	leaq	96(%rsp), %rsi
	leaq	88(%rsp), %rdx
	movq	%r12, %rcx
	movq	%r14, %r8
	movq	%r13, %r9
	callq	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	incq	%r15
	decq	%rbp
	jne	.LBB0_166
.LBB0_167:
	movq	48(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_169
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
.LBB0_169:
	movl	56(%rsp), %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
	movq	168(%rsp), %r15
	movq	160(%rsp), %r14
	movq	128(%rsp), %r12
	movq	136(%rsp), %r13
	movq	176(%rsp), %rbp
.LBB0_189:
	movq	$96, 64(%rsp)
	movq	$11008, 72(%rsp)
	movq	$2048, 80(%rsp)
	movq	%r14, 88(%rsp)
	movq	%r15, 96(%rsp)
	movq	%rbx, 104(%rsp)
	movl	$8454144, %edx
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r13, 112(%rsp)
	movq	%rbp, 120(%rsp)
	testq	%r12, %r12
	jle	.LBB0_239
	cmpq	$1, %r12
	jne	.LBB0_219
	movl	$0, 8(%rsp)
	vstmxcsr	8(%rsp)
	movl	8(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB0_193
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
.LBB0_193:
	movl	%ecx, 56(%rsp)
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_194
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r12
	movq	%rdx, %r15
	leaq	(%rdx,%rdx,2), %rax
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rcx
	movq	%rcx, (%r12,%rax,8)
	movq	$5, 8(%r12,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r12,%rax,8)
	incq	%r15
	jmp	.LBB0_201
.LBB0_219:
	callq	KGEN_CompilerRT_AsyncRT_ParallelismLevel@PLT
	movl	%eax, 136(%rsp)
	movslq	%eax, %r15
	testl	%r15d, %r15d
	movl	$1, %ecx
	cmovneq	%r15, %rcx
	movq	%r12, %rax
	orq	%rcx, %rax
	shrq	$32, %rax
	je	.LBB0_220
	movq	%r12, %rax
	cqto
	idivq	%rcx
	movq	%rax, %r13
	jmp	.LBB0_222
.LBB0_194:
	xorl	%r12d, %r12d
	xorl	%r15d, %r15d
.LBB0_201:
	leaq	static_string_44fd141e40b306d5(%rip), %rdi
	movl	$2, %esi
	movq	%r12, %rdx
	movq	%r15, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r13
	movq	%rcx, %rbp
	xorl	%r14d, %r14d
	testq	%r15, %r15
	cmovgq	%r15, %r14
	movabsq	$4611686018427387904, %rdx
	jle	.LBB0_207
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %r15
	jmp	.LBB0_203
	.p2align	4
.LBB0_206:
	movq	%r15, %rax
	addq	$-1, %rax
	jae	.LBB0_207
.LBB0_203:
	movq	%r14, %rcx
	subq	%r15, %rcx
	movq	%rax, %r15
	leaq	(%rcx,%rcx,2), %rax
	testq	%rdx, 16(%r12,%rax,8)
	je	.LBB0_206
	leaq	(%r12,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB0_206
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movabsq	$4611686018427387904, %rdx
	jmp	.LBB0_206
.LBB0_207:
	movq	%r12, %rdi
	movq	%rdx, %r14
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%r14, %rbp
	je	.LBB0_210
	lock		decq	-8(%r13)
	jne	.LBB0_210
	addq	$-8, %r13
	#MEMBARRIER
	movq	%r13, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB0_210:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB0_211
	leaq	static_string_2b3f504061b33816(%rip), %rdi
	movl	$4, %esi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, 48(%rsp)
	jmp	.LBB0_213
.LBB0_220:
	movl	%r12d, %eax
	xorl	%edx, %edx
	divl	%ecx
	movl	%eax, %r13d
.LBB0_222:
	testl	%r15d, %r15d
	sets	%al
	movq	%rdx, 176(%rsp)
	testq	%rdx, %rdx
	setne	%bpl
	xorl	%r14d, %r14d
	andb	%al, %bpl
	movl	$0, %eax
	cmovneq	%r15, %rax
	movq	%rax, 128(%rsp)
	movq	$0, 144(%rsp)
	leaq	144(%rsp), %rdi
	callq	KGEN_CompilerRT_AsyncRT_InitializeChain@PLT
	movq	$1, 8(%rsp)
	movq	144(%rsp), %rax
	movq	%rax, 16(%rsp)
	movl	$8, %edi
	movl	$128, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, 24(%rsp)
	movq	$0, 32(%rsp)
	movq	$16, 40(%rsp)
	testl	%r15d, %r15d
	je	.LBB0_225
	movzbl	%bpl, %eax
	subq	%rax, %r13
	testq	%r13, %r13
	jle	.LBB0_225
	movq	%r15, %rax
	sarq	$63, %rax
	andnq	%r15, %rax, %r12
	cmpq	$1, %r12
	movq	%r12, %rax
	adcq	$-1, %rax
	movq	%rax, 48(%rsp)
	xorl	%r14d, %r14d
	testl	%r15d, %r15d
	jg	.LBB0_241
.LBB0_225:
	cmpl	$0, 136(%rsp)
	movq	128(%rsp), %r15
	je	.LBB0_231
	addq	176(%rsp), %r15
	testq	%r15, %r15
	jle	.LBB0_231
	leaq	8(%rsp), %rbp
	movq	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"@GOTPCREL(%rip), %r12
	jmp	.LBB0_228
	.p2align	4
.LBB0_230:
	decq	%r15
	movq	%r13, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	testq	%r15, %r15
	je	.LBB0_231
.LBB0_228:
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r13
	movl	$0, (%rax)
	movq	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%r13)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%r13)
	movq	$4, 56(%r13)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%r13)
	movq	$0, 72(%r13)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%r13)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%r13)
	movq	$2, 96(%r13)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%r13)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%r13)
	movq	%r14, 120(%r13)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%r13)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%r13)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%r13)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%r13)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%r13)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%r13)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%r13)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%r13)
	movq	%rbp, 24(%r13)
	movq	%r12, %rdi
	movq	%r13, %rsi
	movq	$-1, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_230
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_230
	.p2align	4
.LBB0_240:
	movq	56(%rsp), %r13
	decq	%r13
	je	.LBB0_225
.LBB0_241:
	movq	%r13, 56(%rsp)
	movq	48(%rsp), %rax
	movq	%r12, %r13
	jmp	.LBB0_242
	.p2align	4
.LBB0_244:
	movq	%rbp, (%rax,%rdx,8)
	incq	32(%rsp)
	incq	%r14
	movq	%r13, %rax
	addq	$-1, %rax
	jae	.LBB0_240
.LBB0_242:
	movq	%r12, %r15
	subq	%r13, %r15
	movq	%rax, %r13
	movl	$8, %edi
	movl	$184, %esi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %rbp
	movl	$0, (%rax)
	movq	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"@GOTPCREL(%rip), %rax
	movq	%rax, 8(%rbp)
	leaq	static_string_2b3f504061b33816(%rip), %rax
	movq	%rax, 48(%rbp)
	movq	$4, 56(%rbp)
	leaq	static_string_c44bdff4074eecdb(%rip), %rax
	movq	%rax, 64(%rbp)
	movq	$0, 72(%rbp)
	leaq	static_string_0c475e2a8e1ec05d(%rip), %rax
	movq	%rax, 80(%rbp)
	leaq	static_string_44fd141e40b306d5(%rip), %rax
	movq	%rax, 88(%rbp)
	movq	$2, 96(%rbp)
	leaq	static_string_f9c5d72f244f07d1(%rip), %rax
	movq	%rax, 104(%rbp)
	leaq	112(%rsp), %rax
	movq	%rax, 112(%rbp)
	movq	%r14, 120(%rbp)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%rbp)
	leaq	96(%rsp), %rax
	movq	%rax, 136(%rbp)
	leaq	88(%rsp), %rax
	movq	%rax, 144(%rbp)
	leaq	104(%rsp), %rax
	movq	%rax, 152(%rbp)
	leaq	72(%rsp), %rax
	movq	%rax, 160(%rbp)
	leaq	80(%rsp), %rax
	movq	%rax, 168(%rbp)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%rbp)
	lock		incq	8(%rsp)
	movq	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"@GOTPCREL(%rip), %rax
	movq	%rax, 16(%rbp)
	leaq	8(%rsp), %rax
	movq	%rax, 24(%rbp)
	movq	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"@GOTPCREL(%rip), %rdi
	movq	%rbp, %rsi
	movq	%r15, %rdx
	callq	KGEN_CompilerRT_AsyncRT_Execute@PLT
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %r8
	cmpq	%r8, %rdx
	jl	.LBB0_244
	xorl	%ecx, %ecx
	testq	%r8, %r8
	sete	%cl
	leaq	(%rcx,%r8,2), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	movq	%r8, %rdx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"@PLT
	movq	%rax, 24(%rsp)
	movq	%rdx, 32(%rsp)
	movq	%rcx, 40(%rsp)
	jmp	.LBB0_244
.LBB0_231:
	leaq	16(%rsp), %r12
	lock		decq	8(%rsp)
	jne	.LBB0_233
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Complete@PLT
.LBB0_233:
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_Wait@PLT
	movq	24(%rsp), %r15
	movq	32(%rsp), %rax
	xorl	%r14d, %r14d
	testq	%rax, %rax
	cmovgq	%rax, %r14
	jle	.LBB0_237
	xorl	%r13d, %r13d
	.p2align	4
.LBB0_235:
	movq	(%r15,%r13,8), %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	incq	%r13
	cmpq	%r13, %r14
	jne	.LBB0_235
	movq	24(%rsp), %r15
.LBB0_237:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r12, %rdi
	callq	KGEN_CompilerRT_AsyncRT_DestroyChain@PLT
	jmp	.LBB0_238
.LBB0_211:
	movq	$0, 48(%rsp)
.LBB0_213:
	movq	120(%rsp), %rax
	xorl	%ebp, %ebp
	testq	%rax, %rax
	setg	%bpl
	addq	112(%rsp), %rbp
	testq	%rbp, %rbp
	jle	.LBB0_216
	movq	%rax, %r15
	sarq	$63, %r15
	andq	%rax, %r15
	leaq	104(%rsp), %r12
	leaq	72(%rsp), %r14
	leaq	80(%rsp), %r13
	.p2align	4
.LBB0_215:
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	movq	%r15, %rdi
	leaq	96(%rsp), %rsi
	leaq	88(%rsp), %rdx
	movq	%r12, %rcx
	movq	%r14, %r8
	movq	%r13, %r9
	callq	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	incq	%r15
	decq	%rbp
	jne	.LBB0_215
.LBB0_216:
	movq	48(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_218
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
.LBB0_218:
	movl	56(%rsp), %eax
	movl	%eax, 8(%rsp)
	vldmxcsr	8(%rsp)
.LBB0_238:
	movq	168(%rsp), %r15
	movq	160(%rsp), %r14
.LBB0_239:
	movq	%r14, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	callq	KGEN_CompilerRT_DestroyGlobals@PLT
	xorl	%eax, %eax
	addq	$200, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.size	.Lmain$local, .Lfunc_end0-main
	.cfi_endproc

	.p2align	4
	.type	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::builtin::simd::SIMD,dtype=f64,size=1\">>, scalar<f64>]",@function
"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::builtin::simd::SIMD,dtype=f64,size=1\">>, scalar<f64>]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, %rbx
	movq	%rsi, %r14
	movq	%rdi, %r15
	leaq	(,%rcx,8), %rsi
	movl	$8, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r12
	movq	%r14, %r13
	shlq	$3, %r13
	cmpq	$4, %r13
	jg	.LBB1_4
	testq	%r13, %r13
	je	.LBB1_12
	movzbl	(%r15), %eax
	movb	%al, (%r12)
	movzbl	-1(%r15,%r13), %eax
	movb	%al, -1(%r12,%r13)
	cmpq	$3, %r13
	jl	.LBB1_13
	movzbl	1(%r15), %eax
	movb	%al, 1(%r12)
	movzbl	-2(%r15,%r13), %eax
	movb	%al, -2(%r12,%r13)
	jmp	.LBB1_13
.LBB1_4:
	cmpq	$16, %r13
	ja	.LBB1_8
	cmpq	$8, %r13
	jl	.LBB1_7
	movq	(%r15), %rax
	movq	%rax, (%r12)
	movq	-8(%r15,%r13), %rax
	movq	%rax, -8(%r12,%r13)
	jmp	.LBB1_13
.LBB1_8:
	movabsq	$9223372036854775776, %rbp
	andq	%r13, %rbp
	je	.LBB1_10
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
.LBB1_10:
	cmpq	%r13, %rbp
	je	.LBB1_12
	movq	%r12, %rdi
	addq	%rbp, %rdi
	addq	%r15, %rbp
	andl	$24, %r13d
	movq	%rbp, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
.LBB1_12:
	testq	%r15, %r15
	jne	.LBB1_13
	jmp	.LBB1_14
.LBB1_7:
	movl	(%r15), %eax
	movl	%eax, (%r12)
	movl	-4(%r15,%r13), %eax
	movl	%eax, -4(%r12,%r13)
.LBB1_13:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB1_14:
	movq	%r12, %rax
	movq	%r14, %rdx
	movq	%rbx, %rcx
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::builtin::simd::SIMD,dtype=f64,size=1\">>, scalar<f64>]", .Lfunc_end1-"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::builtin::simd::SIMD,dtype=f64,size=1\">>, scalar<f64>]"
	.cfi_endproc

	.p2align	4
	.type	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]",@function
"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, %rbx
	movq	%rsi, %r14
	movq	%rdi, %r15
	leaq	(,%rcx,8), %rax
	leaq	(%rax,%rax,2), %rsi
	movl	$8, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r12
	movq	%r14, %rax
	shlq	$3, %rax
	leaq	(%rax,%rax,2), %r13
	cmpq	$4, %r13
	jg	.LBB2_4
	testq	%r13, %r13
	je	.LBB2_12
	movzbl	(%r15), %eax
	movb	%al, (%r12)
	movzbl	-1(%r15,%r13), %eax
	movb	%al, -1(%r12,%r13)
	cmpq	$3, %r13
	jl	.LBB2_13
	movzbl	1(%r15), %eax
	movb	%al, 1(%r12)
	movzbl	-2(%r15,%r13), %eax
	movb	%al, -2(%r12,%r13)
	jmp	.LBB2_13
.LBB2_4:
	cmpq	$16, %r13
	ja	.LBB2_8
	cmpq	$8, %r13
	jl	.LBB2_7
	movq	(%r15), %rax
	movq	%rax, (%r12)
	movq	-8(%r15,%r13), %rax
	movq	%rax, -8(%r12,%r13)
	jmp	.LBB2_13
.LBB2_8:
	movabsq	$9223372036854775776, %rbp
	andq	%r13, %rbp
	je	.LBB2_10
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
.LBB2_10:
	cmpq	%r13, %rbp
	je	.LBB2_12
	movq	%r12, %rdi
	addq	%rbp, %rdi
	addq	%r15, %rbp
	andl	$24, %r13d
	movq	%rbp, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
.LBB2_12:
	testq	%r15, %r15
	jne	.LBB2_13
	jmp	.LBB2_14
.LBB2_7:
	movl	(%r15), %eax
	movl	%eax, (%r12)
	movl	-4(%r15,%r13), %eax
	movl	%eax, -4(%r12,%r13)
.LBB2_13:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB2_14:
	movq	%r12, %rax
	movq	%r14, %rdx
	movq	%rbx, %rcx
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]", .Lfunc_end2-"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"
	.cfi_endproc

	.p2align	4
	.type	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]",@function
"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, %rbx
	movq	%rsi, %r14
	movq	%rdi, %r15
	leaq	(,%rcx,8), %rsi
	movl	$8, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	%rax, %r12
	movq	%r14, %r13
	shlq	$3, %r13
	cmpq	$4, %r13
	jg	.LBB3_4
	testq	%r13, %r13
	je	.LBB3_12
	movzbl	(%r15), %eax
	movb	%al, (%r12)
	movzbl	-1(%r15,%r13), %eax
	movb	%al, -1(%r12,%r13)
	cmpq	$3, %r13
	jl	.LBB3_13
	movzbl	1(%r15), %eax
	movb	%al, 1(%r12)
	movzbl	-2(%r15,%r13), %eax
	movb	%al, -2(%r12,%r13)
	jmp	.LBB3_13
.LBB3_4:
	cmpq	$16, %r13
	ja	.LBB3_8
	cmpq	$8, %r13
	jl	.LBB3_7
	movq	(%r15), %rax
	movq	%rax, (%r12)
	movq	-8(%r15,%r13), %rax
	movq	%rax, -8(%r12,%r13)
	jmp	.LBB3_13
.LBB3_8:
	movabsq	$9223372036854775776, %rbp
	andq	%r13, %rbp
	je	.LBB3_10
	movq	%r12, %rdi
	movq	%r15, %rsi
	movq	%rbp, %rdx
	callq	memcpy@PLT
.LBB3_10:
	cmpq	%r13, %rbp
	je	.LBB3_12
	movq	%r12, %rdi
	addq	%rbp, %rdi
	addq	%r15, %rbp
	andl	$24, %r13d
	movq	%rbp, %rsi
	movq	%r13, %rdx
	callq	memcpy@PLT
.LBB3_12:
	testq	%r15, %r15
	jne	.LBB3_13
	jmp	.LBB3_14
.LBB3_7:
	movl	(%r15), %eax
	movl	%eax, (%r12)
	movl	-4(%r15,%r13), %eax
	movl	%eax, -4(%r12,%r13)
.LBB3_13:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB3_14:
	movq	%r12, %rax
	movq	%r14, %rdx
	movq	%rbx, %rcx
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end3:
	.size	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]", .Lfunc_end3-"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::runtime::asyncrt::_TaskGroupBox\">>, !co.routine]"
	.cfi_endproc

	.p2align	4
	.type	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])",@function
"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r12
	subq	$1, %r12
	jb	.LBB4_33
	movabsq	$4611686018427387904, %r8
	movq	16(%rdi), %r13
	testq	%r13, %r13
	js	.LBB4_9
	movq	8(%rdi), %rbx
	leaq	(%rbx,%rdx), %r15
	leaq	(,%r13,8), %rax
	cmpq	%r8, %r13
	cmovbq	%rbx, %rax
	cmpq	%r15, %rax
	cmovleq	%r15, %rax
	cmpq	$24, %rax
	jge	.LBB4_10
	movq	(%rdi), %r14
	movq	%rbx, %rax
	shlq	$56, %rax
	movabsq	$9223372036854775776, %rcx
	addq	$32, %rcx
	orq	%rax, %rcx
	movq	%rcx, 32(%rsp)
	testq	%rbx, %rbx
	jle	.LBB4_5
	leaq	16(%rsp), %rax
	movq	%rdi, 8(%rsp)
	movq	%rax, %rdi
	movq	%rsi, (%rsp)
	movq	%r14, %rsi
	movq	%rdx, %rbp
	movq	%rbx, %rdx
	callq	memcpy@PLT
	movabsq	$4611686018427387904, %r8
	movq	(%rsp), %rsi
	movq	%rbp, %rdx
	movq	8(%rsp), %rdi
.LBB4_5:
	cmpq	%r8, %r13
	jb	.LBB4_8
	lock		decq	-8(%r14)
	jne	.LBB4_8
	addq	$-8, %r14
	#MEMBARRIER
	movq	%rdi, %r13
	movq	%r14, %rdi
	movq	%rdx, %r14
	movq	%rsi, %rbp
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%rbp, %rsi
	movq	%r14, %rdx
	movq	%r13, %rdi
.LBB4_8:
	movq	16(%rsp), %rax
	movq	%rax, (%rdi)
	movq	32(%rsp), %r13
	vmovups	24(%rsp), %xmm0
	vmovups	%xmm0, 8(%rdi)
	jmp	.LBB4_14
.LBB4_9:
	movq	%r13, %rbx
	shrq	$56, %rbx
	andl	$31, %ebx
	leaq	(%rbx,%rdx), %r15
	movq	%r15, %rax
	movq	%rdi, %rcx
	cmpq	$24, %r15
	jl	.LBB4_17
.LBB4_10:
	testq	%r8, %r13
	je	.LBB4_13
	movq	(%rdi), %rcx
	movq	-8(%rcx), %rcx
	cmpq	$1, %rcx
	jne	.LBB4_13
	leaq	(,%r13,8), %rcx
	testq	%r13, %r13
	movl	$23, %r8d
	cmovnsq	%rcx, %r8
	cmpq	%r8, %rax
	jle	.LBB4_14
.LBB4_13:
	movq	%rdi, %r14
	movq	%rsi, %r13
	movq	%rax, %rsi
	movq	%rdx, %rbp
	callq	"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"@PLT
	movq	%r13, %rsi
	movq	%rbp, %rdx
	movq	%r14, %rdi
	movq	16(%r14), %r13
.LBB4_14:
	testq	%r13, %r13
	js	.LBB4_15
	movq	(%rdi), %rcx
.LBB4_17:
	addq	%rcx, %rbx
	cmpq	$4, %rdx
	jg	.LBB4_20
.LBB4_18:
	movzbl	(%rsi), %eax
	movb	%al, (%rbx)
	movzbl	(%rsi,%r12), %eax
	movb	%al, (%rbx,%r12)
	cmpq	$3, %rdx
	jl	.LBB4_29
	movzbl	1(%rsi), %eax
	movb	%al, 1(%rbx)
	movzbl	-2(%rsi,%rdx), %eax
	movb	%al, -2(%rbx,%rdx)
	movq	16(%rdi), %rax
	testq	%rax, %rax
	jns	.LBB4_31
	jmp	.LBB4_30
.LBB4_15:
	movq	%rdi, %rcx
	addq	%rcx, %rbx
	cmpq	$4, %rdx
	jle	.LBB4_18
.LBB4_20:
	cmpq	$16, %rdx
	jg	.LBB4_24
	cmpq	$8, %rdx
	jl	.LBB4_23
	movq	(%rsi), %rax
	movq	%rax, (%rbx)
	movq	-8(%rsi,%rdx), %rax
	movq	%rax, -8(%rbx,%rdx)
	movq	16(%rdi), %rax
	testq	%rax, %rax
	jns	.LBB4_31
	jmp	.LBB4_30
.LBB4_24:
	movabsq	$9223372036854775776, %rax
	andq	%rdx, %rax
	je	.LBB4_27
	xorl	%ecx, %ecx
	.p2align	4
.LBB4_26:
	vmovups	(%rsi,%rcx), %ymm0
	vmovups	%ymm0, (%rbx,%rcx)
	addq	$32, %rcx
	cmpq	%rax, %rcx
	jb	.LBB4_26
.LBB4_27:
	cmpq	%rdx, %rax
	je	.LBB4_29
	.p2align	4
.LBB4_28:
	movzbl	(%rsi,%rax), %ecx
	movb	%cl, (%rbx,%rax)
	incq	%rax
	cmpq	%rax, %rdx
	jne	.LBB4_28
.LBB4_29:
	movq	16(%rdi), %rax
	testq	%rax, %rax
	js	.LBB4_30
.LBB4_31:
	movq	%r15, 8(%rdi)
	jmp	.LBB4_32
.LBB4_23:
	movl	(%rsi), %eax
	movl	%eax, (%rbx)
	movl	-4(%rsi,%rdx), %eax
	movl	%eax, -4(%rbx,%rdx)
	movq	16(%rdi), %rax
	testq	%rax, %rax
	jns	.LBB4_31
.LBB4_30:
	shlq	$56, %r15
	movabsq	$-2233785415175766017, %rcx
	andq	%rcx, %rax
	orq	%r15, %rax
.LBB4_32:
	movabsq	$-2305843009213693953, %rcx
	andq	%rax, %rcx
	movq	%rcx, 16(%rdi)
.LBB4_33:
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.Lfunc_end4:
	.size	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])", .Lfunc_end4-"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"
	.cfi_endproc

	.p2align	4
	.type	"std::collections::string::string::String::_realloc_mutable(::String&,::Int)",@function
"std::collections::string::string::String::_realloc_mutable(::String&,::Int)":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	movabsq	$4611686018427387904, %rax
	movq	16(%rdi), %r13
	testq	%r13, %r13
	js	.LBB5_1
	movq	(%rbx), %rbp
	movq	8(%rbx), %r15
	leaq	(,%r13,8), %r12
	cmpq	%rax, %r13
	cmovbq	%r15, %r12
	addq	%r12, %r12
	jmp	.LBB5_3
.LBB5_1:
	movq	%r13, %r15
	shrq	$56, %r15
	andl	$31, %r15d
	movl	$46, %r12d
	movq	%rbx, %rbp
.LBB5_3:
	cmpq	%r12, %rsi
	cmovgq	%rsi, %r12
	addq	$7, %r12
	movq	%r12, %rsi
	andq	$-8, %rsi
	addq	$8, %rsi
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	$1, (%rax)
	leaq	8(%rax), %r14
	cmpq	$4, %r15
	jg	.LBB5_7
	testq	%r15, %r15
	je	.LBB5_16
	movzbl	(%rbp), %ecx
	movb	%cl, (%r14)
	movzbl	-1(%rbp,%r15), %ecx
	movb	%cl, -1(%r14,%r15)
	cmpq	$3, %r15
	jl	.LBB5_16
	movzbl	1(%rbp), %ecx
	movb	%cl, 9(%rax)
	movzbl	-2(%rbp,%r15), %ecx
	movb	%cl, 6(%rax,%r15)
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r13
	jne	.LBB5_17
	jmp	.LBB5_19
.LBB5_7:
	cmpq	$16, %r15
	ja	.LBB5_11
	cmpq	$8, %r15
	jl	.LBB5_10
	movq	(%rbp), %rax
	movq	%rax, (%r14)
	movq	-8(%rbp,%r15), %rax
	movq	%rax, -8(%r14,%r15)
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r13
	jne	.LBB5_17
	jmp	.LBB5_19
.LBB5_11:
	movabsq	$9223372036854775776, %rax
	andq	%r15, %rax
	je	.LBB5_14
	xorl	%ecx, %ecx
	.p2align	4
.LBB5_13:
	vmovups	(%rbp,%rcx), %ymm0
	vmovups	%ymm0, (%r14,%rcx)
	addq	$32, %rcx
	cmpq	%rax, %rcx
	jb	.LBB5_13
.LBB5_14:
	cmpq	%r15, %rax
	je	.LBB5_16
	.p2align	4
.LBB5_15:
	movzbl	(%rbp,%rax), %ecx
	movb	%cl, (%r14,%rax)
	incq	%rax
	cmpq	%rax, %r15
	jne	.LBB5_15
.LBB5_16:
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r13
	je	.LBB5_19
.LBB5_17:
	movq	(%rbx), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB5_19
	addq	$-8, %rdi
	#MEMBARRIER
	movq	%rax, %r13
	vzeroupper
	callq	KGEN_CompilerRT_AlignedFree@PLT
	movq	%r13, %rax
.LBB5_19:
	sarq	$3, %r12
	movq	%r15, 8(%rbx)
	movq	%r14, (%rbx)
	orq	%rax, %r12
	movq	%r12, 16(%rbx)
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.LBB5_10:
	.cfi_def_cfa_offset 64
	movl	(%rbp), %eax
	movl	%eax, (%r14)
	movl	-4(%rbp,%r15), %eax
	movl	%eax, -4(%r14,%r15)
	movabsq	$4611686018427387904, %rax
	testq	%rax, %r13
	jne	.LBB5_17
	jmp	.LBB5_19
.Lfunc_end5:
	.size	"std::collections::string::string::String::_realloc_mutable(::String&,::Int)", .Lfunc_end5-"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"
	.cfi_endproc

	.p2align	4
	.type	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0",@function
"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0":
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdx, %rbx
	movq	%rsi, %r15
	movq	%rdi, %r14
	cmpq	$4097, %rdx
	jl	.LBB6_1
	movq	4096(%r14), %rdx
	movq	4104(%r14), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
	movq	$0, 4096(%r14)
	movq	4104(%r14), %rdi
	movq	%r15, %rsi
	movq	%rbx, %rdx
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
.LBB6_1:
	.cfi_def_cfa_offset 48
	movq	4096(%r14), %rdx
	leaq	(%rdx,%rbx), %rax
	cmpq	$4097, %rax
	jl	.LBB6_3
	movq	4104(%r14), %rdi
	movq	%r14, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
	movq	$0, 4096(%r14)
	xorl	%edx, %edx
.LBB6_3:
	addq	%r14, %rdx
	cmpq	$4, %rbx
	jg	.LBB6_7
	testq	%rbx, %rbx
	je	.LBB6_15
	movzbl	(%r15), %eax
	movb	%al, (%rdx)
	movzbl	-1(%r15,%rbx), %eax
	movb	%al, -1(%rdx,%rbx)
	cmpq	$3, %rbx
	jl	.LBB6_15
	movzbl	1(%r15), %eax
	movb	%al, 1(%rdx)
	movzbl	-2(%r15,%rbx), %eax
	movb	%al, -2(%rdx,%rbx)
	jmp	.LBB6_15
.LBB6_7:
	cmpq	$16, %rbx
	jg	.LBB6_11
	cmpq	$8, %rbx
	jl	.LBB6_10
	movq	(%r15), %rax
	movq	%rax, (%rdx)
	movq	-8(%r15,%rbx), %rax
	movq	%rax, -8(%rdx,%rbx)
	jmp	.LBB6_15
.LBB6_11:
	movabsq	$9223372036854775776, %r12
	andq	%rbx, %r12
	je	.LBB6_13
	movq	%rdx, %rdi
	movq	%r15, %rsi
	movq	%rdx, %r13
	movq	%r12, %rdx
	callq	memcpy@PLT
	movq	%r13, %rdx
.LBB6_13:
	cmpq	%rbx, %r12
	je	.LBB6_15
	addq	%r12, %rdx
	addq	%r12, %r15
	movl	%ebx, %eax
	andl	$31, %eax
	movq	%rdx, %rdi
	movq	%r15, %rsi
	movq	%rax, %rdx
	callq	memcpy@PLT
	jmp	.LBB6_15
.LBB6_10:
	movl	(%r15), %eax
	movl	%eax, (%rdx)
	movl	-4(%r15,%rbx), %eax
	movl	%eax, -4(%rdx,%rbx)
.LBB6_15:
	addq	%rbx, 4096(%r14)
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end6:
	.size	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0", .Lfunc_end6-"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"
	.cfi_endproc

	.p2align	4
	.type	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)",@function
"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)":
	.cfi_startproc
	lock		decq	(%rdi)
	jne	.LBB7_1
	addq	$8, %rdi
	jmp	KGEN_CompilerRT_AsyncRT_Complete@PLT
.LBB7_1:
	retq
.Lfunc_end7:
	.size	"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)", .Lfunc_end7-"std::runtime::asyncrt::TaskGroup::_task_complete_callback(::TaskGroup&)"
	.cfi_endproc

	.p2align	4
	.type	"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_0",@function
"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_0":
	.cfi_startproc
	xorl	%edi, %edi
	jmp	KGEN_CompilerRT_AsyncRT_CreateRuntime@PLT
.Lfunc_end8:
	.size	"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_0", .Lfunc_end8-"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_0"
	.cfi_endproc

	.p2align	4
	.type	"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_1",@function
"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_1":
	.cfi_startproc
	jmp	KGEN_CompilerRT_AsyncRT_DestroyRuntime@PLT
.Lfunc_end9:
	.size	"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_1", .Lfunc_end9-"std::builtin::_startup::__wrap_and_execute_main[fn() -> None](::SIMD[::DType(int32), ::Int(1)],!kgen.pointer<pointer<scalar<ui8>>>),main_func=\"asm_driver::main()\"_closure_1"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64",@function
"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64":
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%rcx, -72(%rsp)
	movq	%rdx, -64(%rsp)
	movq	56(%rsp), %rax
	shlq	$5, %rdi
	leaq	32(%rdi), %rcx
	movq	(%rax), %rax
	cmpq	%rax, %rcx
	cmovlq	%rcx, %rax
	cmpq	%rax, %rdi
	movq	%rax, -56(%rsp)
	movq	%rdi, -120(%rsp)
	cmovgq	%rdi, %rax
	movq	%rax, -88(%rsp)
	movq	(%r9), %rcx
	testq	%rcx, %rcx
	jle	.LBB10_4
	movq	%r8, %r11
	xorl	%edi, %edi
	movq	-120(%rsp), %rdx
	cmpq	-56(%rsp), %rdx
	setl	%al
	movq	(%r8), %r8
	testq	%r8, %r8
	jle	.LBB10_2
	movq	%rcx, -40(%rsp)
	movb	%al, %dil
	orq	%rdx, %rdi
	movq	%rdi, -112(%rsp)
	xorl	%ebp, %ebp
	.p2align	4
.LBB10_7:
	leaq	32(%rbp), %rcx
	movq	(%r9), %rax
	cmpq	%rax, %rcx
	movq	%rcx, -32(%rsp)
	cmovlq	%rcx, %rax
	cmpq	%rax, %rbp
	movq	%rax, %r13
	cmovgq	%rbp, %r13
	testq	%r8, %r8
	jle	.LBB10_5
	cmpq	%rax, %rbp
	setl	%dil
	movq	-56(%rsp), %rcx
	cmpq	%rcx, -120(%rsp)
	jge	.LBB10_9
	cmpq	%rax, %rbp
	jge	.LBB10_12
	xorl	%eax, %eax
	movb	%dil, %al
	orq	%rbp, %rax
	movq	%rax, -80(%rsp)
	movl	$32, %eax
	movq	%rax, -96(%rsp)
	movq	$0, -104(%rsp)
	xorl	%ebx, %ebx
	xorl	%eax, %eax
	movq	%rbp, -48(%rsp)
	movq	%r8, -24(%rsp)
	jmp	.LBB10_15
	.p2align	4
.LBB10_23:
	addq	$256, %rbx
	addq	$32, -96(%rsp)
	addq	$-32, -104(%rsp)
	movq	-24(%rsp), %r8
	movq	-16(%rsp), %rax
	cmpq	%r8, %rax
	jge	.LBB10_5
.LBB10_15:
	leaq	32(%rax), %rcx
	movq	(%r11), %r15
	cmpq	%r15, %rcx
	movq	%r15, %rdi
	movq	%rcx, -16(%rsp)
	cmovlq	%rcx, %rdi
	subq	%rax, %rdi
	movq	%rdi, %r14
	andq	$-8, %r14
	cmpq	$7, %rdi
	jle	.LBB10_16
	movq	-112(%rsp), %r8
	movq	-120(%rsp), %r10
	cmpq	%rdi, %r14
	jne	.LBB10_25
	.p2align	4
.LBB10_33:
	movq	%r8, %rax
	leaq	(,%r10,8), %r14
	movq	-80(%rsp), %r8
	movq	%rbp, %r15
	movq	-72(%rsp), %rdx
	.p2align	4
.LBB10_34:
	movq	%r15, %r12
	movq	%r8, %r15
	movq	(%r9), %r8
	imulq	%r10, %r8
	shlq	$3, %r8
	addq	(%rdx), %r8
	movq	(%r11), %rcx
	vbroadcastsd	(%r8,%r12,8), %zmm0
	shlq	$3, %r12
	movq	(%rsi), %r8
	addq	%rbx, %r8
	imulq	%rcx, %r12
	addq	%r8, %r12
	movq	-64(%rsp), %r8
	movq	(%r8), %r8
	addq	%rbx, %r8
	imulq	%r14, %rcx
	addq	%r8, %rcx
	xorl	%r8d, %r8d
	.p2align	4
.LBB10_35:
	vmovupd	(%r12,%r8,8), %zmm1
	vfmadd213pd	(%rcx,%r8,8), %zmm0, %zmm1
	vmovupd	%zmm1, (%rcx,%r8,8)
	addq	$8, %r8
	cmpq	%rdi, %r8
	jl	.LBB10_35
	xorl	%r8d, %r8d
	cmpq	%r13, %r15
	setne	%r8b
	addq	%r15, %r8
	cmpq	%r13, %r15
	jne	.LBB10_34
	xorl	%r8d, %r8d
	movq	-88(%rsp), %rcx
	cmpq	%rcx, %rax
	setne	%r8b
	addq	%rax, %r8
	movq	%rax, %r10
	cmpq	%rcx, %rax
	jne	.LBB10_33
	jmp	.LBB10_23
	.p2align	4
.LBB10_16:
	cmpq	%rdi, %r14
	je	.LBB10_23
	movq	-96(%rsp), %rax
	cmpq	%rax, %r15
	cmovgeq	%rax, %r15
	addq	-104(%rsp), %r15
	movq	%r15, %rax
	andq	$-8, %rax
	cmpq	%r15, %rax
	cmovleq	%r15, %rax
	movq	-112(%rsp), %r8
	movq	-120(%rsp), %r10
	.p2align	4
.LBB10_18:
	movq	%r8, %rdi
	leaq	(,%r10,8), %r15
	movq	-80(%rsp), %r8
	movq	%rbp, %r12
	movq	-72(%rsp), %rdx
	.p2align	4
.LBB10_19:
	movq	%r12, %rbp
	movq	%r8, %r12
	movq	(%r9), %r8
	imulq	%r10, %r8
	shlq	$3, %r8
	addq	(%rdx), %r8
	vmovsd	(%r8,%rbp,8), %xmm0
	shlq	$3, %rbp
	movq	(%r11), %rcx
	movq	(%rsi), %r8
	addq	%rbx, %r8
	imulq	%rcx, %rbp
	addq	%r8, %rbp
	movq	-64(%rsp), %r8
	movq	(%r8), %r8
	addq	%rbx, %r8
	imulq	%r15, %rcx
	addq	%r8, %rcx
	movq	%r14, %r8
	.p2align	4
.LBB10_20:
	vmovsd	(%rbp,%r8,8), %xmm1
	vfmadd213sd	(%rcx,%r8,8), %xmm0, %xmm1
	vmovsd	%xmm1, (%rcx,%r8,8)
	incq	%r8
	cmpq	%r8, %rax
	jne	.LBB10_20
	xorl	%r8d, %r8d
	cmpq	%r13, %r12
	setne	%r8b
	addq	%r12, %r8
	cmpq	%r13, %r12
	jne	.LBB10_19
	xorl	%r8d, %r8d
	movq	-88(%rsp), %rcx
	cmpq	%rcx, %rdi
	setne	%r8b
	addq	%rdi, %r8
	movq	%rdi, %r10
	cmpq	%rcx, %rdi
	movq	-48(%rsp), %rbp
	jne	.LBB10_18
	jmp	.LBB10_23
	.p2align	4
.LBB10_25:
	movq	-96(%rsp), %rax
	cmpq	%rax, %r15
	cmovgeq	%rax, %r15
	addq	-104(%rsp), %r15
	movq	%r15, %rdi
	andq	$-8, %rdi
	cmpq	%r15, %rdi
	cmovleq	%r15, %rdi
	movq	-112(%rsp), %rax
	movq	-120(%rsp), %rdx
	.p2align	4
.LBB10_26:
	movq	%rax, -8(%rsp)
	movq	-80(%rsp), %rax
	movq	%rbp, %r12
	.p2align	4
.LBB10_27:
	movq	%rax, %rbp
	movq	%r9, %r8
	movq	(%r9), %r9
	imulq	%rdx, %r9
	shlq	$3, %r9
	movq	-72(%rsp), %rax
	addq	(%rax), %r9
	leaq	(,%r12,8), %rax
	vbroadcastsd	(%r9,%r12,8), %zmm0
	movq	%r11, %rcx
	movq	(%r11), %r11
	movq	(%rsi), %r15
	addq	%rbx, %r15
	imulq	%r11, %rax
	addq	%r15, %rax
	movq	-64(%rsp), %r9
	movq	(%r9), %r9
	addq	%rbx, %r9
	leaq	(,%rdx,8), %r10
	imulq	%r11, %r10
	addq	%r9, %r10
	xorl	%r9d, %r9d
	.p2align	4
.LBB10_28:
	vmovupd	(%rax,%r9,8), %zmm1
	vfmadd213pd	(%r10,%r9,8), %zmm0, %zmm1
	vmovupd	%zmm1, (%r10,%r9,8)
	addq	$8, %r9
	cmpq	%r14, %r9
	jl	.LBB10_28
	imulq	%r11, %r12
	leaq	(%r15,%r12,8), %rax
	movq	%r14, %r11
	.p2align	4
.LBB10_30:
	vmovsd	(%rax,%r11,8), %xmm1
	vfmadd213sd	(%r10,%r11,8), %xmm0, %xmm1
	vmovsd	%xmm1, (%r10,%r11,8)
	incq	%r11
	cmpq	%r11, %rdi
	jne	.LBB10_30
	xorl	%eax, %eax
	cmpq	%r13, %rbp
	setne	%al
	addq	%rbp, %rax
	movq	%rbp, %r12
	cmpq	%r13, %rbp
	movq	%r8, %r9
	movq	%rcx, %r11
	jne	.LBB10_27
	xorl	%eax, %eax
	movq	-88(%rsp), %rcx
	movq	-8(%rsp), %r8
	cmpq	%rcx, %r8
	setne	%al
	addq	%r8, %rax
	movq	%r8, %rdx
	cmpq	%rcx, %r8
	movq	-48(%rsp), %rbp
	jne	.LBB10_26
	jmp	.LBB10_23
	.p2align	4
.LBB10_9:
	xorl	%eax, %eax
	.p2align	4
.LBB10_10:
	addq	$32, %rax
	cmpq	%r8, %rax
	jl	.LBB10_10
	jmp	.LBB10_5
.LBB10_12:
	xorl	%eax, %eax
	.p2align	4
.LBB10_13:
	addq	$32, %rax
	cmpq	%r8, %rax
	jl	.LBB10_13
	.p2align	4
.LBB10_5:
	movq	-32(%rsp), %rax
	cmpq	-40(%rsp), %rax
	jge	.LBB10_4
	movq	(%r11), %r8
	movq	%rax, %rbp
	jmp	.LBB10_7
.LBB10_2:
	xorl	%eax, %eax
	.p2align	4
.LBB10_3:
	addq	$32, %rax
	cmpq	%rcx, %rax
	jl	.LBB10_3
.LBB10_4:
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end10:
	.size	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64", .Lfunc_end10-"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"

	.p2align	4
	.type	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume",@function
"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB11_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB11_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB11_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB11_5
.LBB11_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB11_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB11_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB11_7
	.p2align	4
.LBB11_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB11_11
.LBB11_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB11_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB11_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB11_10
.LBB11_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB11_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB11_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB11_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB11_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB11_17
.LBB11_15:
	xorl	%ebx, %ebx
.LBB11_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB11_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB11_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB11_19
.LBB11_20:
	testq	%rbx, %rbx
	je	.LBB11_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB11_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end11:
	.size	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume", .Lfunc_end11-"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1",@function
"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end12:
	.size	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1", .Lfunc_end12-"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume",@function
"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB13_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB13_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB13_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB13_5
.LBB13_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB13_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB13_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB13_7
	.p2align	4
.LBB13_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB13_11
.LBB13_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB13_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB13_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB13_10
.LBB13_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB13_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB13_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB13_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB13_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB13_17
.LBB13_15:
	xorl	%ebx, %ebx
.LBB13_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB13_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB13_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB13_19
.LBB13_20:
	testq	%rbx, %rbx
	je	.LBB13_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB13_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end13:
	.size	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume", .Lfunc_end13-"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3",@function
"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end14:
	.size	"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3", .Lfunc_end14-"gemm::matmul_parallel[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64",@function
"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64":
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$136, %rsp
	movq	%r8, -120(%rsp)
	movq	%rcx, -56(%rsp)
	movq	%rdx, -64(%rsp)
	movq	%rsi, -72(%rsp)
	movq	192(%rsp), %rax
	movq	%rdi, %rdx
	shlq	$5, %rdi
	movq	%rdi, -32(%rsp)
	leaq	32(%rdi), %rcx
	movq	(%rax), %rax
	cmpq	%rax, %rcx
	cmovlq	%rcx, %rax
	movq	%rax, -112(%rsp)
	movq	%r9, -104(%rsp)
	movq	(%r9), %rcx
	testq	%rcx, %rcx
	jle	.LBB15_4
	movq	-120(%rsp), %rax
	cmpq	$0, (%rax)
	jle	.LBB15_2
	movq	-32(%rsp), %rax
	orq	$4, %rax
	movq	%rax, 24(%rsp)
	shlq	$8, %rdx
	leaq	24(%rdx), %rax
	movq	%rax, 64(%rsp)
	leaq	16(%rdx), %rax
	movq	%rax, 56(%rsp)
	movq	%rdx, 72(%rsp)
	leaq	8(%rdx), %rax
	movq	%rax, 48(%rsp)
	xorl	%eax, %eax
	movq	%rcx, 32(%rsp)
	jmp	.LBB15_6
	.p2align	4
.LBB15_5:
	movq	32(%rsp), %rcx
	movq	40(%rsp), %rax
	cmpq	%rcx, %rax
	jge	.LBB15_4
.LBB15_6:
	movq	%rax, %rdx
	leaq	32(%rax), %rcx
	movq	-104(%rsp), %rax
	movq	(%rax), %rax
	cmpq	%rax, %rcx
	movq	%rcx, 40(%rsp)
	cmovlq	%rcx, %rax
	cmpq	%rax, %rdx
	movq	%rax, -40(%rsp)
	movq	%rdx, -128(%rsp)
	cmovgq	%rdx, %rax
	movq	%rax, -48(%rsp)
	movq	-120(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 80(%rsp)
	testq	%rax, %rax
	jle	.LBB15_5
	xorl	%ecx, %ecx
	movq	-128(%rsp), %rax
	cmpq	-40(%rsp), %rax
	setl	%cl
	orq	%rax, %rcx
	movq	%rcx, -80(%rsp)
	movl	$32, %eax
	movq	%rax, -88(%rsp)
	movq	$0, -96(%rsp)
	xorl	%r14d, %r14d
	xorl	%eax, %eax
	jmp	.LBB15_9
	.p2align	4
.LBB15_8:
	addq	$256, %r14
	addq	$32, -88(%rsp)
	addq	$-32, -96(%rsp)
	movq	88(%rsp), %rax
	cmpq	80(%rsp), %rax
	jge	.LBB15_5
.LBB15_9:
	movq	%r14, -16(%rsp)
	leaq	32(%rax), %rdx
	movq	-120(%rsp), %rcx
	movq	(%rcx), %r10
	cmpq	%r10, %rdx
	movq	%r10, -24(%rsp)
	movq	%rdx, 88(%rsp)
	cmovlq	%rdx, %r10
	subq	%rax, %r10
	movq	-32(%rsp), %r15
	movq	24(%rsp), %rax
	cmpq	-112(%rsp), %rax
	jle	.LBB15_10
.LBB15_19:
	cmpq	-112(%rsp), %r15
	movq	-16(%rsp), %r14
	jge	.LBB15_8
	movq	-40(%rsp), %rax
	cmpq	%rax, -128(%rsp)
	jge	.LBB15_8
	movq	%r10, %rbx
	andq	$-8, %rbx
	cmpq	$8, %r10
	jl	.LBB15_39
	cmpq	%r10, %rbx
	jne	.LBB15_23
	leaq	(,%r15,8), %rax
	.p2align	4
.LBB15_34:
	movq	-80(%rsp), %rsi
	movq	-128(%rsp), %rcx
	.p2align	4
.LBB15_35:
	movq	%rcx, %rdx
	movq	%rsi, %rcx
	movq	-104(%rsp), %rsi
	movq	(%rsi), %rdi
	imulq	%r15, %rdi
	shlq	$3, %rdi
	movq	-56(%rsp), %rsi
	addq	(%rsi), %rdi
	movq	-120(%rsp), %rsi
	movq	(%rsi), %rsi
	vbroadcastsd	(%rdi,%rdx,8), %zmm0
	shlq	$3, %rdx
	movq	-72(%rsp), %rdi
	movq	(%rdi), %rdi
	addq	%r14, %rdi
	imulq	%rsi, %rdx
	addq	%rdi, %rdx
	movq	-64(%rsp), %rdi
	movq	(%rdi), %rdi
	addq	%r14, %rdi
	imulq	%rax, %rsi
	addq	%rdi, %rsi
	xorl	%edi, %edi
	.p2align	4
.LBB15_36:
	vmovupd	(%rdx,%rdi,8), %zmm1
	vfmadd213pd	(%rsi,%rdi,8), %zmm0, %zmm1
	vmovupd	%zmm1, (%rsi,%rdi,8)
	addq	$8, %rdi
	cmpq	%r10, %rdi
	jl	.LBB15_36
	xorl	%esi, %esi
	movq	-48(%rsp), %rdx
	cmpq	%rdx, %rcx
	setne	%sil
	addq	%rcx, %rsi
	cmpq	%rdx, %rcx
	jne	.LBB15_35
	incq	%r15
	addq	$8, %rax
	cmpq	-112(%rsp), %r15
	jne	.LBB15_34
	jmp	.LBB15_8
	.p2align	4
.LBB15_10:
	movq	-40(%rsp), %rax
	cmpq	%rax, -128(%rsp)
	movq	-16(%rsp), %r14
	jge	.LBB15_8
	movq	%r10, %rbx
	andq	$-8, %rbx
	movq	-88(%rsp), %r13
	movq	-24(%rsp), %rax
	cmpq	%r13, %rax
	cmovlq	%rax, %r13
	addq	-96(%rsp), %r13
	movq	%r13, %rax
	andq	$-8, %rax
	leaq	(%r14,%rax,8), %rcx
	movq	%rcx, 104(%rsp)
	cmpq	%r13, %rax
	cmovgq	%rax, %r13
	subq	%rax, %r13
	movq	72(%rsp), %rax
	movq	%rax, 16(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	56(%rsp), %rax
	movq	%rax, (%rsp)
	movq	64(%rsp), %rax
	movq	%rax, -8(%rsp)
	movq	24(%rsp), %rax
	movq	-32(%rsp), %r8
	jmp	.LBB15_12
	.p2align	4
.LBB15_18:
	movq	96(%rsp), %r15
	leaq	4(%r15), %rax
	addq	$32, -8(%rsp)
	addq	$32, (%rsp)
	addq	$32, 8(%rsp)
	addq	$32, 16(%rsp)
	movq	%r15, %r8
	cmpq	-112(%rsp), %rax
	jg	.LBB15_19
.LBB15_12:
	movq	%rax, 96(%rsp)
	movq	%r8, %rax
	orq	$3, %rax
	movq	%rax, 128(%rsp)
	movq	%r8, %rax
	orq	$2, %rax
	movq	%rax, 120(%rsp)
	movq	%r8, %rax
	orq	$1, %rax
	movq	%rax, 112(%rsp)
	movq	-80(%rsp), %rax
	movq	-128(%rsp), %rdx
	jmp	.LBB15_13
	.p2align	4
.LBB15_17:
	xorl	%eax, %eax
	movq	-48(%rsp), %rcx
	cmpq	%rcx, %rsi
	setne	%al
	addq	%rsi, %rax
	movq	%rsi, %rdx
	cmpq	%rcx, %rsi
	movq	%rdi, %r8
	je	.LBB15_18
.LBB15_13:
	movq	%rax, %rsi
	movq	-104(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, %rcx
	movq	%r8, %rdi
	imulq	%r8, %rcx
	movq	-56(%rsp), %r8
	movq	(%r8), %r8
	leaq	(%r8,%rcx,8), %rcx
	vmovsd	(%rcx,%rdx,8), %xmm0
	movq	%rax, %rcx
	imulq	112(%rsp), %rcx
	leaq	(%r8,%rcx,8), %rcx
	vmovsd	(%rcx,%rdx,8), %xmm1
	movq	%rax, %rcx
	imulq	120(%rsp), %rcx
	imulq	128(%rsp), %rax
	leaq	(%r8,%rcx,8), %rcx
	vmovsd	(%rcx,%rdx,8), %xmm2
	leaq	(%r8,%rax,8), %rax
	vmovsd	(%rax,%rdx,8), %xmm3
	movq	-120(%rsp), %rax
	movq	(%rax), %r15
	movq	-72(%rsp), %rax
	movq	(%rax), %r12
	movq	-64(%rsp), %rax
	movq	(%rax), %rax
	cmpq	$7, %r10
	jle	.LBB15_14
	vbroadcastsd	%xmm0, %zmm4
	vbroadcastsd	%xmm1, %zmm5
	vbroadcastsd	%xmm2, %zmm6
	vbroadcastsd	%xmm3, %zmm7
	movq	-16(%rsp), %r9
	leaq	(%rax,%r9), %rcx
	movq	-8(%rsp), %r14
	imulq	%r15, %r14
	addq	%rcx, %r14
	movq	(%rsp), %rbp
	imulq	%r15, %rbp
	addq	%rcx, %rbp
	movq	8(%rsp), %r11
	imulq	%r15, %r11
	addq	%rcx, %r11
	movq	16(%rsp), %r8
	imulq	%r15, %r8
	addq	%rcx, %r8
	leaq	(%r12,%r9), %rcx
	movq	%rdx, %r9
	imulq	%r15, %r9
	leaq	(%rcx,%r9,8), %rcx
	xorl	%r9d, %r9d
	.p2align	4
.LBB15_32:
	vmovupd	(%rcx,%r9,8), %zmm8
	vmovupd	(%r8,%r9,8), %zmm9
	vfmadd231pd	%zmm8, %zmm4, %zmm9
	vmovupd	%zmm9, (%r8,%r9,8)
	vmovupd	(%r11,%r9,8), %zmm9
	vfmadd231pd	%zmm8, %zmm5, %zmm9
	vmovupd	%zmm9, (%r11,%r9,8)
	vmovupd	(%rbp,%r9,8), %zmm9
	vfmadd231pd	%zmm8, %zmm6, %zmm9
	vmovupd	%zmm9, (%rbp,%r9,8)
	vfmadd213pd	(%r14,%r9,8), %zmm7, %zmm8
	vmovupd	%zmm8, (%r14,%r9,8)
	addq	$8, %r9
	cmpq	%rbx, %r9
	jl	.LBB15_32
.LBB15_14:
	cmpq	%r10, %rbx
	je	.LBB15_17
	movq	104(%rsp), %rcx
	addq	%rcx, %rax
	movq	-8(%rsp), %r14
	imulq	%r15, %r14
	addq	%rax, %r14
	movq	(%rsp), %rbp
	imulq	%r15, %rbp
	addq	%rax, %rbp
	movq	8(%rsp), %r11
	imulq	%r15, %r11
	addq	%rax, %r11
	movq	16(%rsp), %r8
	imulq	%r15, %r8
	addq	%rax, %r8
	addq	%rcx, %r12
	imulq	%r15, %rdx
	leaq	(%r12,%rdx,8), %rax
	xorl	%ecx, %ecx
	.p2align	4
.LBB15_16:
	vmovsd	(%rax,%rcx,8), %xmm4
	vmovsd	(%r8,%rcx,8), %xmm5
	vfmadd231sd	%xmm0, %xmm4, %xmm5
	vmovsd	%xmm5, (%r8,%rcx,8)
	vmovsd	(%r11,%rcx,8), %xmm5
	vfmadd231sd	%xmm1, %xmm4, %xmm5
	vmovsd	%xmm5, (%r11,%rcx,8)
	vmovsd	(%rbp,%rcx,8), %xmm5
	vfmadd231sd	%xmm2, %xmm4, %xmm5
	vmovsd	%xmm5, (%rbp,%rcx,8)
	vfmadd213sd	(%r14,%rcx,8), %xmm3, %xmm4
	vmovsd	%xmm4, (%r14,%rcx,8)
	incq	%rcx
	cmpq	%rcx, %r13
	jne	.LBB15_16
	jmp	.LBB15_17
.LBB15_39:
	cmpq	%r10, %rbx
	je	.LBB15_8
	leaq	(,%r15,8), %rcx
	movq	-88(%rsp), %rax
	movq	-24(%rsp), %rdx
	cmpq	%rax, %rdx
	cmovgeq	%rax, %rdx
	addq	-96(%rsp), %rdx
	movq	%rdx, %rax
	andq	$-8, %rax
	cmpq	%rdx, %rax
	cmovleq	%rdx, %rax
	.p2align	4
.LBB15_41:
	movq	-80(%rsp), %rdi
	movq	-128(%rsp), %rdx
	.p2align	4
.LBB15_42:
	movq	%rdx, %rsi
	movq	%rdi, %rdx
	movq	-104(%rsp), %rdi
	movq	(%rdi), %rdi
	imulq	%r15, %rdi
	shlq	$3, %rdi
	movq	-56(%rsp), %r8
	addq	(%r8), %rdi
	vmovsd	(%rdi,%rsi,8), %xmm0
	shlq	$3, %rsi
	movq	-120(%rsp), %rdi
	movq	(%rdi), %rdi
	movq	-72(%rsp), %r8
	movq	(%r8), %r8
	addq	%r14, %r8
	imulq	%rdi, %rsi
	addq	%r8, %rsi
	movq	-64(%rsp), %r8
	movq	(%r8), %r8
	addq	%r14, %r8
	imulq	%rcx, %rdi
	addq	%r8, %rdi
	movq	%rbx, %r8
	.p2align	4
.LBB15_43:
	vmovsd	(%rsi,%r8,8), %xmm1
	vfmadd213sd	(%rdi,%r8,8), %xmm0, %xmm1
	vmovsd	%xmm1, (%rdi,%r8,8)
	incq	%r8
	cmpq	%r8, %rax
	jne	.LBB15_43
	xorl	%edi, %edi
	movq	-48(%rsp), %rsi
	cmpq	%rsi, %rdx
	setne	%dil
	addq	%rdx, %rdi
	cmpq	%rsi, %rdx
	jne	.LBB15_42
	incq	%r15
	addq	$8, %rcx
	cmpq	-112(%rsp), %r15
	jne	.LBB15_41
	jmp	.LBB15_8
.LBB15_23:
	leaq	(,%r15,8), %rcx
	movq	-88(%rsp), %rax
	movq	-24(%rsp), %rsi
	cmpq	%rax, %rsi
	cmovgeq	%rax, %rsi
	addq	-96(%rsp), %rsi
	movq	%rsi, %rdx
	andq	$-8, %rdx
	cmpq	%rsi, %rdx
	cmovleq	%rsi, %rdx
	.p2align	4
.LBB15_24:
	movq	-80(%rsp), %rdi
	movq	-128(%rsp), %rsi
	.p2align	4
.LBB15_25:
	movq	%rdi, %rax
	movq	-104(%rsp), %rdi
	movq	(%rdi), %rdi
	imulq	%r15, %rdi
	shlq	$3, %rdi
	movq	-56(%rsp), %r8
	addq	(%r8), %rdi
	leaq	(,%rsi,8), %r8
	vbroadcastsd	(%rdi,%rsi,8), %zmm0
	movq	-120(%rsp), %rdi
	movq	(%rdi), %r9
	movq	-72(%rsp), %rdi
	movq	(%rdi), %r10
	addq	%r14, %r10
	imulq	%r9, %r8
	addq	%r10, %r8
	movq	-64(%rsp), %rdi
	movq	(%rdi), %r11
	addq	%r14, %r11
	movq	%rcx, %rdi
	imulq	%r9, %rdi
	addq	%r11, %rdi
	xorl	%r11d, %r11d
	.p2align	4
.LBB15_26:
	vmovupd	(%r8,%r11,8), %zmm1
	vfmadd213pd	(%rdi,%r11,8), %zmm0, %zmm1
	vmovupd	%zmm1, (%rdi,%r11,8)
	addq	$8, %r11
	cmpq	%rbx, %r11
	jl	.LBB15_26
	imulq	%r9, %rsi
	leaq	(%r10,%rsi,8), %rsi
	movq	%rbx, %r8
	.p2align	4
.LBB15_28:
	vmovsd	(%rsi,%r8,8), %xmm1
	vfmadd213sd	(%rdi,%r8,8), %xmm0, %xmm1
	vmovsd	%xmm1, (%rdi,%r8,8)
	incq	%r8
	cmpq	%r8, %rdx
	jne	.LBB15_28
	xorl	%edi, %edi
	movq	-48(%rsp), %r8
	cmpq	%r8, %rax
	setne	%dil
	addq	%rax, %rdi
	movq	%rax, %rsi
	cmpq	%r8, %rax
	jne	.LBB15_25
	incq	%r15
	addq	$8, %rcx
	cmpq	-112(%rsp), %r15
	jne	.LBB15_24
	jmp	.LBB15_8
.LBB15_2:
	xorl	%eax, %eax
	.p2align	4
.LBB15_3:
	addq	$32, %rax
	cmpq	%rcx, %rax
	jl	.LBB15_3
.LBB15_4:
	addq	$136, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end15:
	.size	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64", .Lfunc_end15-"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"

	.p2align	4
	.type	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume",@function
"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB16_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB16_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB16_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB16_5
.LBB16_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB16_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB16_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB16_7
	.p2align	4
.LBB16_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB16_11
.LBB16_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB16_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB16_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB16_10
.LBB16_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB16_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB16_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB16_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB16_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB16_17
.LBB16_15:
	xorl	%ebx, %ebx
.LBB16_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB16_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB16_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB16_19
.LBB16_20:
	testq	%rbx, %rbx
	je	.LBB16_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB16_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end16:
	.size	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume", .Lfunc_end16-"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1",@function
"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end17:
	.size	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1", .Lfunc_end17-"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume",@function
"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB18_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB18_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB18_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB18_5
.LBB18_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB18_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB18_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB18_7
	.p2align	4
.LBB18_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB18_11
.LBB18_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB18_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB18_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB18_10
.LBB18_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB18_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB18_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB18_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB18_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB18_17
.LBB18_15:
	xorl	%ebx, %ebx
.LBB18_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB18_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB18_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB18_19
.LBB18_20:
	testq	%rbx, %rbx
	je	.LBB18_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB18_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end18:
	.size	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume", .Lfunc_end18-"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3",@function
"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end19:
	.size	"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3", .Lfunc_end19-"gemm::matmul_register_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64",@function
"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64":
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$96, %rsp
	movq	%rcx, -88(%rsp)
	movq	%rsi, -96(%rsp)
	movq	152(%rsp), %rax
	movq	%rdi, %rsi
	shlq	$5, %rdi
	movq	%rdi, -32(%rsp)
	leaq	32(%rdi), %rcx
	movq	(%rax), %r13
	cmpq	%r13, %rcx
	cmovlq	%rcx, %r13
	movq	(%r9), %rcx
	testq	%rcx, %rcx
	jle	.LBB20_4
	movq	(%r8), %rdi
	testq	%rdi, %rdi
	jle	.LBB20_2
	movq	%r9, %r12
	movq	%rcx, -8(%rsp)
	movq	-32(%rsp), %rax
	orq	$4, %rax
	movq	%rax, 32(%rsp)
	shlq	$8, %rsi
	leaq	24(%rsi), %rax
	movq	%rax, 24(%rsp)
	leaq	16(%rsi), %rax
	movq	%rax, 16(%rsp)
	movq	%rsi, 40(%rsp)
	leaq	8(%rsi), %rax
	movq	%rax, 8(%rsp)
	movl	$32, %eax
	movq	%rax, -40(%rsp)
	movq	$0, -104(%rsp)
	movq	$0, -48(%rsp)
	xorl	%eax, %eax
	movq	%r8, 88(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%r13, 64(%rsp)
	.p2align	4
.LBB20_7:
	leaq	32(%rax), %rsi
	movq	(%r12), %rcx
	cmpq	%rcx, %rsi
	movq	%rcx, %r9
	movq	%rsi, (%rsp)
	cmovlq	%rsi, %r9
	movq	%r9, -112(%rsp)
	testq	%rdi, %rdi
	jle	.LBB20_5
	subq	%rax, -112(%rsp)
	movq	-40(%rsp), %rax
	cmpq	%rax, %rcx
	cmovgeq	%rax, %rcx
	addq	-48(%rsp), %rcx
	movq	%rcx, %rax
	sarq	$63, %rax
	andnq	%rcx, %rax, %r15
	xorl	%ecx, %ecx
	xorl	%eax, %eax
	movq	%rdi, 48(%rsp)
	jmp	.LBB20_10
	.p2align	4
.LBB20_9:
	movq	-120(%rsp), %rcx
	addq	$256, %rcx
	movq	48(%rsp), %rdi
	movq	56(%rsp), %rax
	cmpq	%rdi, %rax
	jge	.LBB20_5
.LBB20_10:
	movq	%rcx, -120(%rsp)
	movq	%rax, %rcx
	addq	$32, %rax
	movq	(%r8), %rsi
	cmpq	%rsi, %rax
	movq	%rax, 56(%rsp)
	cmovlq	%rax, %rsi
	movq	%rcx, -24(%rsp)
	subq	%rcx, %rsi
	movq	%rsi, -128(%rsp)
	movq	-32(%rsp), %rbp
	movq	40(%rsp), %rax
	movq	%rax, -56(%rsp)
	movq	8(%rsp), %rax
	movq	%rax, -64(%rsp)
	movq	16(%rsp), %rax
	movq	%rax, -72(%rsp)
	movq	24(%rsp), %rax
	movq	%rax, -80(%rsp)
	movq	32(%rsp), %rax
	movq	%rax, %rcx
	cmpq	%r13, %rax
	jle	.LBB20_11
.LBB20_15:
	cmpq	%r13, %rbp
	jge	.LBB20_9
	leaq	(,%rbp,8), %r14
	jmp	.LBB20_17
	.p2align	4
.LBB20_14:
	movq	80(%rsp), %rbp
	leaq	4(%rbp), %rcx
	addq	$32, -80(%rsp)
	addq	$32, -72(%rsp)
	addq	$32, -64(%rsp)
	addq	$32, -56(%rsp)
	movq	64(%rsp), %r13
	cmpq	%r13, %rcx
	movq	72(%rsp), %rdx
	jg	.LBB20_15
.LBB20_11:
	movq	(%r8), %r10
	movq	%r10, %rsi
	imulq	%rbp, %rsi
	movq	%rbp, %rdi
	orq	$1, %rdi
	imulq	%r10, %rdi
	movq	%rbp, %r9
	orq	$2, %r9
	imulq	%r10, %r9
	orq	$3, %rbp
	imulq	%r10, %rbp
	movq	(%rdx), %rdx
	leaq	(%rdx,%rbp,8), %rax
	movq	%rcx, 80(%rsp)
	leaq	(%rdx,%rsi,8), %rcx
	movq	-24(%rsp), %rsi
	leaq	(%rcx,%rsi,8), %r10
	leaq	(%rdx,%rdi,8), %rcx
	leaq	(%rcx,%rsi,8), %rbp
	leaq	(%rdx,%r9,8), %rcx
	leaq	(%rcx,%rsi,8), %r13
	leaq	(%rax,%rsi,8), %r14
	cmpq	$8, -128(%rsp)
	movq	%r10, -16(%rsp)
	jge	.LBB20_24
	xorl	%ebx, %ebx
	jmp	.LBB20_13
	.p2align	4
.LBB20_24:
	cmpq	$0, -112(%rsp)
	jle	.LBB20_29
	movl	$8, %eax
	movq	-120(%rsp), %r11
	xorl	%ecx, %ecx
	.p2align	4
.LBB20_26:
	movq	%rax, %rbx
	vmovupd	(%r10,%rcx,8), %zmm3
	vmovupd	(%rbp,%rcx,8), %zmm2
	vmovupd	(%r13,%rcx,8), %zmm1
	vmovupd	(%r14,%rcx,8), %zmm0
	movq	(%r8), %rsi
	movq	%r12, %r9
	movq	(%r12), %r8
	movq	-88(%rsp), %rax
	movq	(%rax), %rax
	movq	-104(%rsp), %rdi
	addq	%rdi, %rax
	movq	-80(%rsp), %rdx
	imulq	%r8, %rdx
	addq	%rax, %rdx
	movq	-72(%rsp), %r12
	imulq	%r8, %r12
	addq	%rax, %r12
	movq	-64(%rsp), %r10
	imulq	%r8, %r10
	addq	%rax, %r10
	imulq	-56(%rsp), %r8
	addq	%rax, %r8
	movq	%rdi, %rax
	imulq	%rsi, %rax
	addq	%r11, %rax
	movq	-96(%rsp), %rdi
	addq	(%rdi), %rax
	shlq	$3, %rsi
	xorl	%edi, %edi
	.p2align	4
.LBB20_27:
	vmovupd	(%rax), %zmm4
	vfmadd231pd	(%r8,%rdi,8){1to8}, %zmm4, %zmm3
	vfmadd231pd	(%r10,%rdi,8){1to8}, %zmm4, %zmm2
	vfmadd231pd	(%r12,%rdi,8){1to8}, %zmm4, %zmm1
	vfmadd231pd	(%rdx,%rdi,8){1to8}, %zmm4, %zmm0
	incq	%rdi
	addq	%rsi, %rax
	cmpq	%rdi, %r15
	jne	.LBB20_27
	movq	-16(%rsp), %r10
	vmovupd	%zmm3, (%r10,%rcx,8)
	vmovupd	%zmm2, (%rbp,%rcx,8)
	vmovupd	%zmm1, (%r13,%rcx,8)
	vmovupd	%zmm0, (%r14,%rcx,8)
	leaq	8(%rbx), %rax
	addq	$64, %r11
	movq	%rbx, %rcx
	cmpq	-128(%rsp), %rax
	movq	%r9, %r12
	movq	88(%rsp), %r8
	jle	.LBB20_26
.LBB20_13:
	cmpq	$0, -112(%rsp)
	setle	%al
	cmpq	-128(%rsp), %rbx
	setge	%cl
	orb	%al, %cl
	jne	.LBB20_14
	movq	-120(%rsp), %rax
	leaq	(%rax,%rbx,8), %r11
	.p2align	4
.LBB20_32:
	vmovsd	(%r10,%rbx,8), %xmm3
	vmovsd	(%rbp,%rbx,8), %xmm2
	vmovsd	(%r13,%rbx,8), %xmm1
	vmovsd	(%r14,%rbx,8), %xmm0
	movq	(%r8), %rcx
	movq	(%r12), %rsi
	movq	-88(%rsp), %rax
	movq	(%rax), %rax
	movq	-104(%rsp), %rdi
	addq	%rdi, %rax
	movq	-80(%rsp), %rdx
	imulq	%rsi, %rdx
	addq	%rax, %rdx
	movq	-72(%rsp), %r9
	imulq	%rsi, %r9
	addq	%rax, %r9
	movq	-64(%rsp), %r10
	imulq	%rsi, %r10
	addq	%rax, %r10
	imulq	-56(%rsp), %rsi
	addq	%rax, %rsi
	movq	%rdi, %rax
	imulq	%rcx, %rax
	addq	%r11, %rax
	movq	-96(%rsp), %rdi
	addq	(%rdi), %rax
	shlq	$3, %rcx
	xorl	%edi, %edi
	.p2align	4
.LBB20_33:
	vmovsd	(%rax), %xmm4
	vfmadd231sd	(%rsi,%rdi,8), %xmm4, %xmm3
	vfmadd231sd	(%r10,%rdi,8), %xmm4, %xmm2
	vfmadd231sd	(%r9,%rdi,8), %xmm4, %xmm1
	vfmadd231sd	(%rdx,%rdi,8), %xmm4, %xmm0
	incq	%rdi
	addq	%rcx, %rax
	cmpq	%rdi, %r15
	jne	.LBB20_33
	movq	-16(%rsp), %r10
	vmovsd	%xmm3, (%r10,%rbx,8)
	vmovsd	%xmm2, (%rbp,%rbx,8)
	vmovsd	%xmm1, (%r13,%rbx,8)
	vmovsd	%xmm0, (%r14,%rbx,8)
	incq	%rbx
	addq	$8, %r11
	cmpq	-128(%rsp), %rbx
	jl	.LBB20_32
	jmp	.LBB20_14
.LBB20_29:
	movl	$8, %eax
	.p2align	4
.LBB20_30:
	addq	$8, %rax
	cmpq	-128(%rsp), %rax
	jle	.LBB20_30
	jmp	.LBB20_14
	.p2align	4
.LBB20_42:
	incq	%rbp
	addq	$8, %r14
	cmpq	%r13, %rbp
	je	.LBB20_9
.LBB20_17:
	movq	(%r8), %rax
	imulq	%rbp, %rax
	shlq	$3, %rax
	addq	(%rdx), %rax
	movq	-24(%rsp), %rcx
	leaq	(%rax,%rcx,8), %rbx
	cmpq	$8, -128(%rsp)
	jge	.LBB20_35
	xorl	%r10d, %r10d
	jmp	.LBB20_19
	.p2align	4
.LBB20_35:
	cmpq	$0, -112(%rsp)
	jle	.LBB20_40
	movl	$8, %eax
	movq	-120(%rsp), %rcx
	xorl	%esi, %esi
	.p2align	4
.LBB20_37:
	movq	%rax, %r10
	vmovupd	(%rbx,%rsi,8), %zmm0
	movq	(%r8), %rax
	movq	-88(%rsp), %rdi
	movq	(%rdi), %rdi
	movq	-104(%rsp), %r11
	addq	%r11, %rdi
	movq	(%r12), %r9
	imulq	%r14, %r9
	addq	%rdi, %r9
	movq	%r11, %rdi
	imulq	%rax, %rdi
	addq	%rcx, %rdi
	movq	-96(%rsp), %r11
	addq	(%r11), %rdi
	shlq	$3, %rax
	xorl	%r11d, %r11d
	.p2align	4
.LBB20_38:
	vmovupd	(%rdi), %zmm1
	vfmadd231pd	(%r9,%r11,8){1to8}, %zmm1, %zmm0
	incq	%r11
	addq	%rax, %rdi
	cmpq	%r11, %r15
	jne	.LBB20_38
	vmovupd	%zmm0, (%rbx,%rsi,8)
	leaq	8(%r10), %rax
	addq	$64, %rcx
	movq	%r10, %rsi
	cmpq	-128(%rsp), %rax
	jle	.LBB20_37
.LBB20_19:
	cmpq	$0, -112(%rsp)
	setle	%al
	cmpq	-128(%rsp), %r10
	setge	%cl
	orb	%al, %cl
	jne	.LBB20_42
	movq	-120(%rsp), %rax
	leaq	(%rax,%r10,8), %rcx
	.p2align	4
.LBB20_21:
	vmovsd	(%rbx,%r10,8), %xmm0
	movq	(%r8), %rax
	movq	-104(%rsp), %r11
	movq	%r11, %rsi
	imulq	%rax, %rsi
	addq	%rcx, %rsi
	movq	-96(%rsp), %rdi
	addq	(%rdi), %rsi
	shlq	$3, %rax
	movq	-88(%rsp), %rdi
	movq	(%rdi), %r9
	addq	%r11, %r9
	movq	(%r12), %rdi
	imulq	%r14, %rdi
	addq	%r9, %rdi
	xorl	%r9d, %r9d
	.p2align	4
.LBB20_22:
	vmovsd	(%rdi,%r9,8), %xmm1
	vfmadd231sd	(%rsi), %xmm1, %xmm0
	addq	%rax, %rsi
	incq	%r9
	cmpq	%r9, %r15
	jne	.LBB20_22
	vmovsd	%xmm0, (%rbx,%r10,8)
	incq	%r10
	addq	$8, %rcx
	cmpq	-128(%rsp), %r10
	jl	.LBB20_21
	jmp	.LBB20_42
.LBB20_40:
	movl	$8, %eax
	.p2align	4
.LBB20_41:
	addq	$8, %rax
	cmpq	-128(%rsp), %rax
	jle	.LBB20_41
	jmp	.LBB20_42
	.p2align	4
.LBB20_5:
	movq	(%rsp), %rax
	cmpq	-8(%rsp), %rax
	jge	.LBB20_4
	movq	(%r8), %rdi
	addq	$256, -104(%rsp)
	addq	$32, -40(%rsp)
	addq	$-32, -48(%rsp)
	jmp	.LBB20_7
.LBB20_2:
	xorl	%eax, %eax
	.p2align	4
.LBB20_3:
	addq	$32, %rax
	cmpq	%rcx, %rax
	jl	.LBB20_3
.LBB20_4:
	addq	$96, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end20:
	.size	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64", .Lfunc_end20-"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"

	.p2align	4
	.type	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume",@function
"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB21_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB21_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB21_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB21_5
.LBB21_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB21_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB21_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB21_7
	.p2align	4
.LBB21_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB21_11
.LBB21_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB21_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB21_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB21_10
.LBB21_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB21_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB21_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB21_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB21_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB21_17
.LBB21_15:
	xorl	%ebx, %ebx
.LBB21_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB21_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB21_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB21_19
.LBB21_20:
	testq	%rbx, %rbx
	je	.LBB21_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB21_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end21:
	.size	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume", .Lfunc_end21-"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1",@function
"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end22:
	.size	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1", .Lfunc_end22-"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume",@function
"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB23_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB23_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB23_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB23_5
.LBB23_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB23_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB23_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB23_7
	.p2align	4
.LBB23_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB23_11
.LBB23_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB23_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB23_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB23_10
.LBB23_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB23_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB23_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB23_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB23_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB23_17
.LBB23_15:
	xorl	%ebx, %ebx
.LBB23_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB23_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB23_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB23_19
.LBB23_20:
	testq	%rbx, %rbx
	je	.LBB23_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB23_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end23:
	.size	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume", .Lfunc_end23-"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3",@function
"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end24:
	.size	"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3", .Lfunc_end24-"gemm::matmul_packed[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64",@function
"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64":
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$104, %rsp
	movq	%rcx, -88(%rsp)
	movq	%rsi, -96(%rsp)
	movq	160(%rsp), %rax
	movq	%rdi, %rsi
	shlq	$5, %rdi
	movq	%rdi, (%rsp)
	leaq	32(%rdi), %rcx
	movq	(%rax), %r13
	cmpq	%r13, %rcx
	cmovlq	%rcx, %r13
	movq	(%r9), %rcx
	testq	%rcx, %rcx
	jle	.LBB25_4
	movq	(%r8), %rdi
	testq	%rdi, %rdi
	jle	.LBB25_2
	movq	%rcx, 8(%rsp)
	movq	(%rsp), %rax
	orq	$4, %rax
	movq	%rax, 48(%rsp)
	shlq	$8, %rsi
	leaq	24(%rsi), %rax
	movq	%rax, 40(%rsp)
	leaq	16(%rsi), %rax
	movq	%rax, 32(%rsp)
	movq	%rsi, 56(%rsp)
	leaq	8(%rsi), %rax
	movq	%rax, 24(%rsp)
	movl	$32, %eax
	movq	%rax, -8(%rsp)
	movq	$0, -112(%rsp)
	movq	$0, -16(%rsp)
	xorl	%eax, %eax
	movq	%r8, -32(%rsp)
	movq	%r9, -72(%rsp)
	movq	%rdx, 88(%rsp)
	movq	%r13, 80(%rsp)
	.p2align	4
.LBB25_7:
	leaq	32(%rax), %rsi
	movq	(%r9), %rcx
	cmpq	%rcx, %rsi
	movq	%rcx, %r10
	movq	%rsi, 16(%rsp)
	cmovlq	%rsi, %r10
	movq	%r10, -120(%rsp)
	testq	%rdi, %rdi
	jle	.LBB25_5
	subq	%rax, -120(%rsp)
	movq	-8(%rsp), %rax
	cmpq	%rax, %rcx
	cmovgeq	%rax, %rcx
	addq	-16(%rsp), %rcx
	movq	%rcx, %rax
	sarq	$63, %rax
	andnq	%rcx, %rax, %r15
	movq	$0, -128(%rsp)
	xorl	%eax, %eax
	movq	%rdi, 64(%rsp)
	jmp	.LBB25_10
	.p2align	4
.LBB25_9:
	addq	$256, -128(%rsp)
	movq	64(%rsp), %rdi
	movq	72(%rsp), %rax
	cmpq	%rdi, %rax
	jge	.LBB25_5
.LBB25_10:
	movq	%rax, %rbp
	addq	$32, %rax
	movq	(%r8), %rbx
	cmpq	%rbx, %rax
	movq	%rax, 72(%rsp)
	cmovlq	%rax, %rbx
	subq	%rbp, %rbx
	movq	(%rsp), %r14
	movq	56(%rsp), %rax
	movq	%rax, -40(%rsp)
	movq	24(%rsp), %rax
	movq	%rax, -48(%rsp)
	movq	32(%rsp), %rax
	movq	%rax, -56(%rsp)
	movq	40(%rsp), %rax
	movq	%rax, -64(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, %rax
	cmpq	%r13, %rcx
	movq	%rbx, -104(%rsp)
	movq	%rbp, 96(%rsp)
	jle	.LBB25_11
.LBB25_16:
	cmpq	%r13, %r14
	jge	.LBB25_9
	leaq	(,%r14,8), %rcx
	jmp	.LBB25_18
	.p2align	4
.LBB25_15:
	movq	-80(%rsp), %r14
	leaq	4(%r14), %rax
	addq	$32, -64(%rsp)
	addq	$32, -56(%rsp)
	addq	$32, -48(%rsp)
	addq	$32, -40(%rsp)
	movq	80(%rsp), %r13
	cmpq	%r13, %rax
	movq	-32(%rsp), %r8
	movq	88(%rsp), %rdx
	movq	96(%rsp), %rbp
	jg	.LBB25_16
.LBB25_11:
	movq	%r14, %rcx
	movq	(%r8), %r10
	movq	%r10, %rsi
	imulq	%r14, %rsi
	movq	%r14, %rdi
	orq	$1, %rdi
	imulq	%r10, %rdi
	movq	%r14, %r9
	orq	$2, %r9
	imulq	%r10, %r9
	orq	$3, %rcx
	imulq	%r10, %rcx
	movq	(%rdx), %rdx
	leaq	(%rdx,%rcx,8), %rcx
	movq	%rax, -80(%rsp)
	leaq	(%rdx,%rsi,8), %rax
	leaq	(%rax,%rbp,8), %r10
	leaq	(%rdx,%rdi,8), %rax
	leaq	(%rax,%rbp,8), %r13
	leaq	(%rdx,%r9,8), %rax
	leaq	(%rax,%rbp,8), %r14
	leaq	(%rcx,%rbp,8), %rbp
	cmpq	$32, %rbx
	movq	%r10, -24(%rsp)
	jge	.LBB25_30
	xorl	%ecx, %ecx
.LBB25_13:
	movq	%rcx, %rdx
	orq	$8, %rdx
	movq	-104(%rsp), %rbx
	cmpq	%rbx, %rdx
	jle	.LBB25_25
	movq	%rcx, %r12
	movq	-72(%rsp), %r9
	jmp	.LBB25_36
	.p2align	4
.LBB25_30:
	movl	$32, %eax
	movq	-128(%rsp), %r9
	xorl	%ebx, %ebx
	jmp	.LBB25_31
	.p2align	4
.LBB25_34:
	movq	%rbx, %rax
	orq	$8, %rax
	movq	%rbx, %rdx
	orq	$16, %rdx
	movq	%rbx, %rsi
	orq	$24, %rsi
	movq	-24(%rsp), %r10
	vmovupd	%zmm0, (%r10,%rbx,8)
	vmovupd	%zmm1, (%r10,%rax,8)
	vmovupd	%zmm2, (%r10,%rdx,8)
	vmovupd	%zmm3, (%r10,%rsi,8)
	vmovupd	%zmm4, (%r13,%rbx,8)
	vmovupd	%zmm5, (%r13,%rax,8)
	vmovupd	%zmm6, (%r13,%rdx,8)
	vmovupd	%zmm7, (%r13,%rsi,8)
	vmovupd	%zmm8, (%r14,%rbx,8)
	vmovupd	%zmm9, (%r14,%rax,8)
	vmovupd	%zmm10, (%r14,%rdx,8)
	vmovupd	%zmm11, (%r14,%rsi,8)
	vmovupd	%zmm12, (%rbp,%rbx,8)
	vmovupd	%zmm13, (%rbp,%rax,8)
	vmovupd	%zmm14, (%rbp,%rdx,8)
	vmovupd	%zmm15, (%rbp,%rsi,8)
	leaq	32(%rcx), %rax
	addq	$256, %r9
	movq	%rcx, %rbx
	cmpq	-104(%rsp), %rax
	jg	.LBB25_13
.LBB25_31:
	vmovupd	(%r10,%rbx,8), %zmm0
	vmovupd	64(%r10,%rbx,8), %zmm1
	vmovupd	128(%r10,%rbx,8), %zmm2
	vmovupd	192(%r10,%rbx,8), %zmm3
	vmovupd	(%r13,%rbx,8), %zmm4
	vmovupd	64(%r13,%rbx,8), %zmm5
	vmovupd	128(%r13,%rbx,8), %zmm6
	vmovupd	192(%r13,%rbx,8), %zmm7
	vmovupd	(%r14,%rbx,8), %zmm8
	vmovupd	64(%r14,%rbx,8), %zmm9
	vmovupd	128(%r14,%rbx,8), %zmm10
	vmovupd	192(%r14,%rbx,8), %zmm11
	vmovupd	(%rbp,%rbx,8), %zmm12
	vmovupd	64(%rbp,%rbx,8), %zmm13
	movq	%rax, %rcx
	vmovupd	128(%rbp,%rbx,8), %zmm14
	vmovupd	192(%rbp,%rbx,8), %zmm15
	cmpq	$0, -120(%rsp)
	jle	.LBB25_34
	movq	(%r8), %rsi
	movq	-72(%rsp), %rax
	movq	(%rax), %r10
	movq	-88(%rsp), %rax
	movq	(%rax), %rax
	movq	-112(%rsp), %r12
	addq	%r12, %rax
	movq	-64(%rsp), %rdx
	imulq	%r10, %rdx
	addq	%rax, %rdx
	movq	-56(%rsp), %r11
	imulq	%r10, %r11
	addq	%rax, %r11
	movq	-48(%rsp), %rdi
	imulq	%r10, %rdi
	addq	%rax, %rdi
	imulq	-40(%rsp), %r10
	addq	%rax, %r10
	movq	%r12, %rax
	imulq	%rsi, %rax
	addq	%r9, %rax
	movq	-96(%rsp), %r12
	addq	(%r12), %rax
	shlq	$3, %rsi
	xorl	%r12d, %r12d
	.p2align	4
.LBB25_33:
	vmovupd	(%rax), %zmm16
	vmovupd	64(%rax), %zmm17
	vmovupd	128(%rax), %zmm18
	vmovupd	192(%rax), %zmm19
	vbroadcastsd	(%r10,%r12,8), %zmm20
	vfmadd231pd	%zmm20, %zmm16, %zmm0
	vfmadd231pd	%zmm20, %zmm17, %zmm1
	vfmadd231pd	%zmm20, %zmm18, %zmm2
	vfmadd231pd	%zmm20, %zmm19, %zmm3
	vbroadcastsd	(%rdi,%r12,8), %zmm20
	vfmadd231pd	%zmm20, %zmm16, %zmm4
	vfmadd231pd	%zmm20, %zmm17, %zmm5
	vfmadd231pd	%zmm20, %zmm18, %zmm6
	vfmadd231pd	%zmm20, %zmm19, %zmm7
	vbroadcastsd	(%r11,%r12,8), %zmm20
	vfmadd231pd	%zmm20, %zmm16, %zmm8
	vfmadd231pd	%zmm20, %zmm17, %zmm9
	vfmadd231pd	%zmm20, %zmm18, %zmm10
	vfmadd231pd	%zmm20, %zmm19, %zmm11
	vbroadcastsd	(%rdx,%r12,8), %zmm20
	vfmadd231pd	%zmm16, %zmm20, %zmm12
	vfmadd231pd	%zmm17, %zmm20, %zmm13
	vfmadd231pd	%zmm18, %zmm20, %zmm14
	vfmadd231pd	%zmm20, %zmm19, %zmm15
	incq	%r12
	addq	%rsi, %rax
	cmpq	%r12, %r15
	jne	.LBB25_33
	jmp	.LBB25_34
	.p2align	4
.LBB25_25:
	cmpq	$0, -120(%rsp)
	movq	-72(%rsp), %r9
	jle	.LBB25_35
	movq	-128(%rsp), %rax
	leaq	(%rax,%rcx,8), %rax
	.p2align	4
.LBB25_27:
	movq	%rdx, %r12
	vmovupd	(%r10,%rcx,8), %zmm3
	vmovupd	(%r13,%rcx,8), %zmm2
	vmovupd	(%r14,%rcx,8), %zmm1
	vmovupd	(%rbp,%rcx,8), %zmm0
	movq	-32(%rsp), %rdx
	movq	(%rdx), %rdx
	movq	(%r9), %rsi
	movq	-88(%rsp), %rdi
	movq	(%rdi), %r11
	movq	-112(%rsp), %r9
	addq	%r9, %r11
	movq	-64(%rsp), %rdi
	imulq	%rsi, %rdi
	addq	%r11, %rdi
	movq	-56(%rsp), %r8
	imulq	%rsi, %r8
	addq	%r11, %r8
	movq	-48(%rsp), %r10
	imulq	%rsi, %r10
	addq	%r11, %r10
	imulq	-40(%rsp), %rsi
	addq	%r11, %rsi
	movq	%r9, %r11
	imulq	%rdx, %r11
	addq	%rax, %r11
	movq	-96(%rsp), %r9
	addq	(%r9), %r11
	shlq	$3, %rdx
	xorl	%ebx, %ebx
	.p2align	4
.LBB25_28:
	vmovupd	(%r11), %zmm4
	vfmadd231pd	(%rsi,%rbx,8){1to8}, %zmm4, %zmm3
	vfmadd231pd	(%r10,%rbx,8){1to8}, %zmm4, %zmm2
	vfmadd231pd	(%r8,%rbx,8){1to8}, %zmm4, %zmm1
	vfmadd231pd	(%rdi,%rbx,8){1to8}, %zmm4, %zmm0
	incq	%rbx
	addq	%rdx, %r11
	cmpq	%rbx, %r15
	jne	.LBB25_28
	movq	-24(%rsp), %r10
	vmovupd	%zmm3, (%r10,%rcx,8)
	vmovupd	%zmm2, (%r13,%rcx,8)
	vmovupd	%zmm1, (%r14,%rcx,8)
	vmovupd	%zmm0, (%rbp,%rcx,8)
	leaq	8(%r12), %rdx
	addq	$64, %rax
	movq	%r12, %rcx
	movq	-104(%rsp), %rbx
	cmpq	%rbx, %rdx
	movq	-72(%rsp), %r9
	jle	.LBB25_27
	jmp	.LBB25_36
	.p2align	4
.LBB25_35:
	leaq	8(%rcx), %r12
	addq	$16, %rcx
	cmpq	%rbx, %rcx
	movq	%r12, %rcx
	jle	.LBB25_35
	.p2align	4
.LBB25_36:
	cmpq	%rbx, %r12
	jge	.LBB25_15
	movq	-128(%rsp), %rax
	leaq	(%rax,%r12,8), %rax
	jmp	.LBB25_38
	.p2align	4
.LBB25_41:
	movq	-24(%rsp), %r10
	vmovsd	%xmm0, (%r10,%r12,8)
	vmovsd	%xmm1, (%r13,%r12,8)
	vmovsd	%xmm2, (%r14,%r12,8)
	vmovsd	%xmm3, (%rbp,%r12,8)
	incq	%r12
	addq	$8, %rax
	cmpq	%rbx, %r12
	jge	.LBB25_15
.LBB25_38:
	vmovsd	(%r10,%r12,8), %xmm0
	vmovsd	(%r13,%r12,8), %xmm1
	vmovsd	(%r14,%r12,8), %xmm2
	vmovsd	(%rbp,%r12,8), %xmm3
	cmpq	$0, -120(%rsp)
	jle	.LBB25_41
	movq	-32(%rsp), %rcx
	movq	(%rcx), %rcx
	movq	(%r9), %rdx
	movq	-88(%rsp), %rsi
	movq	(%rsi), %r10
	movq	-112(%rsp), %r11
	addq	%r11, %r10
	movq	-64(%rsp), %rsi
	imulq	%rdx, %rsi
	addq	%r10, %rsi
	movq	-56(%rsp), %rdi
	imulq	%rdx, %rdi
	addq	%r10, %rdi
	movq	-48(%rsp), %r8
	imulq	%rdx, %r8
	addq	%r10, %r8
	imulq	-40(%rsp), %rdx
	addq	%r10, %rdx
	movq	%r11, %r10
	imulq	%rcx, %r10
	addq	%rax, %r10
	movq	-96(%rsp), %r11
	addq	(%r11), %r10
	shlq	$3, %rcx
	xorl	%r11d, %r11d
	.p2align	4
.LBB25_40:
	vmovsd	(%r10), %xmm4
	vfmadd231sd	(%rdx,%r11,8), %xmm4, %xmm0
	vfmadd231sd	(%r8,%r11,8), %xmm4, %xmm1
	vfmadd231sd	(%rdi,%r11,8), %xmm4, %xmm2
	vfmadd231sd	(%rsi,%r11,8), %xmm4, %xmm3
	incq	%r11
	addq	%rcx, %r10
	cmpq	%r11, %r15
	jne	.LBB25_40
	jmp	.LBB25_41
	.p2align	4
.LBB25_49:
	incq	%r14
	addq	$8, %rcx
	cmpq	%r13, %r14
	je	.LBB25_9
.LBB25_18:
	movq	(%r8), %rax
	imulq	%r14, %rax
	shlq	$3, %rax
	addq	(%rdx), %rax
	leaq	(%rax,%rbp,8), %r12
	cmpq	$8, %rbx
	jge	.LBB25_42
	movq	%r14, -80(%rsp)
	xorl	%eax, %eax
	jmp	.LBB25_20
	.p2align	4
.LBB25_42:
	cmpq	$0, -120(%rsp)
	jle	.LBB25_47
	movq	%r14, -80(%rsp)
	movl	$8, %r10d
	movq	-128(%rsp), %rsi
	xorl	%edi, %edi
	.p2align	4
.LBB25_44:
	movq	%r10, %rax
	vmovupd	(%r12,%rdi,8), %zmm0
	movq	(%r8), %r14
	movq	-88(%rsp), %r10
	movq	(%r10), %r10
	movq	-112(%rsp), %r11
	addq	%r11, %r10
	movq	(%r9), %rbx
	imulq	%rcx, %rbx
	addq	%r10, %rbx
	movq	%r11, %r10
	imulq	%r14, %r10
	addq	%rsi, %r10
	movq	-96(%rsp), %r11
	addq	(%r11), %r10
	shlq	$3, %r14
	xorl	%r11d, %r11d
	.p2align	4
.LBB25_45:
	vmovupd	(%r10), %zmm1
	vfmadd231pd	(%rbx,%r11,8){1to8}, %zmm1, %zmm0
	incq	%r11
	addq	%r14, %r10
	cmpq	%r11, %r15
	jne	.LBB25_45
	vmovupd	%zmm0, (%r12,%rdi,8)
	leaq	8(%rax), %r10
	addq	$64, %rsi
	movq	%rax, %rdi
	movq	-104(%rsp), %rbx
	cmpq	%rbx, %r10
	jle	.LBB25_44
.LBB25_20:
	cmpq	$0, -120(%rsp)
	setle	%sil
	cmpq	%rbx, %rax
	setge	%dil
	orb	%sil, %dil
	movq	-80(%rsp), %r14
	jne	.LBB25_49
	movq	-128(%rsp), %rsi
	leaq	(%rsi,%rax,8), %rsi
	.p2align	4
.LBB25_22:
	vmovsd	(%r12,%rax,8), %xmm0
	movq	(%r8), %rdi
	movq	-112(%rsp), %r11
	movq	%r11, %rbx
	imulq	%rdi, %rbx
	addq	%rsi, %rbx
	movq	-96(%rsp), %r10
	addq	(%r10), %rbx
	shlq	$3, %rdi
	movq	-88(%rsp), %r10
	movq	(%r10), %r10
	addq	%r11, %r10
	movq	(%r9), %r11
	imulq	%rcx, %r11
	addq	%r10, %r11
	xorl	%r10d, %r10d
	.p2align	4
.LBB25_23:
	vmovsd	(%r11,%r10,8), %xmm1
	vfmadd231sd	(%rbx), %xmm1, %xmm0
	addq	%rdi, %rbx
	incq	%r10
	cmpq	%r10, %r15
	jne	.LBB25_23
	vmovsd	%xmm0, (%r12,%rax,8)
	incq	%rax
	addq	$8, %rsi
	movq	-104(%rsp), %rbx
	cmpq	%rbx, %rax
	jl	.LBB25_22
	jmp	.LBB25_49
.LBB25_47:
	movl	$8, %eax
	.p2align	4
.LBB25_48:
	addq	$8, %rax
	cmpq	%rbx, %rax
	jle	.LBB25_48
	jmp	.LBB25_49
	.p2align	4
.LBB25_5:
	movq	16(%rsp), %rax
	cmpq	8(%rsp), %rax
	jge	.LBB25_4
	movq	(%r8), %rdi
	addq	$256, -112(%rsp)
	addq	$32, -8(%rsp)
	addq	$-32, -16(%rsp)
	jmp	.LBB25_7
.LBB25_2:
	xorl	%eax, %eax
	.p2align	4
.LBB25_3:
	addq	$32, %rax
	cmpq	%rcx, %rax
	jl	.LBB25_3
.LBB25_4:
	addq	$104, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end25:
	.size	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64", .Lfunc_end25-"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"

	.p2align	4
	.type	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume",@function
"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB26_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB26_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB26_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB26_5
.LBB26_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB26_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB26_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB26_7
	.p2align	4
.LBB26_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB26_11
.LBB26_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB26_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB26_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB26_10
.LBB26_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB26_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB26_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB26_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB26_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB26_17
.LBB26_15:
	xorl	%ebx, %ebx
.LBB26_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB26_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB26_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB26_19
.LBB26_20:
	testq	%rbx, %rbx
	je	.LBB26_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB26_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end26:
	.size	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume", .Lfunc_end26-"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_0_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1",@function
"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end27:
	.size	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1", .Lfunc_end27-"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_1"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume",@function
"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$56, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$0, 16(%rsp)
	vstmxcsr	16(%rsp)
	movl	16(%rsp), %ecx
	movl	%ecx, %eax
	notl	%eax
	testl	$32832, %eax
	je	.LBB28_2
	movl	%ecx, %eax
	orl	$32832, %eax
	movl	%eax, 24(%rsp)
	vldmxcsr	24(%rsp)
.LBB28_2:
	movl	%ecx, 20(%rsp)
	movq	48(%rdi), %rax
	movq	%rax, 48(%rsp)
	movq	56(%rdi), %rax
	movq	%rax, 40(%rsp)
	movq	%rdi, %rbx
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB28_3
	movl	$1, %ecx
	xorl	%edi, %edi
	xorl	%esi, %esi
	xorl	%edx, %edx
	callq	"std::collections::list::List::_realloc(::List[$0]&,::Int),T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, %r15
	movq	%rdx, %rbp
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rbx), %rcx
	movq	%rcx, (%r15,%rax,8)
	movq	$5, 8(%r15,%rax,8)
	movabsq	$2305843009213693952, %rcx
	movq	%rcx, 16(%r15,%rax,8)
	incq	%rbp
	jmp	.LBB28_5
.LBB28_3:
	xorl	%r15d, %r15d
	xorl	%ebp, %ebp
.LBB28_5:
	movq	%rbx, %rax
	movabsq	$4611686018427387904, %rbx
	movq	88(%rax), %rdi
	movq	%rax, %r12
	movq	96(%rax), %rsi
	movq	%r15, %rdx
	movq	%rbp, %rcx
	callq	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"@PLT
	movq	%rax, 32(%rsp)
	movq	%rcx, %r13
	xorl	%r14d, %r14d
	testq	%rbp, %rbp
	cmovgq	%rbp, %r14
	jle	.LBB28_11
	cmpq	$1, %r14
	movq	%r14, %rax
	adcq	$-1, %rax
	movq	%r14, %rbp
	jmp	.LBB28_7
	.p2align	4
.LBB28_10:
	movq	%rbp, %rax
	addq	$-1, %rax
	jae	.LBB28_11
.LBB28_7:
	movq	%r14, %rcx
	subq	%rbp, %rcx
	movq	%rax, %rbp
	leaq	(%rcx,%rcx,2), %rax
	testq	%rbx, 16(%r15,%rax,8)
	je	.LBB28_10
	leaq	(%r15,%rax,8), %rax
	movq	(%rax), %rdi
	lock		decq	-8(%rdi)
	jne	.LBB28_10
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
	jmp	.LBB28_10
.LBB28_11:
	movq	%r15, %rdi
	callq	KGEN_CompilerRT_AlignedFree@PLT
	testq	%rbx, %r13
	je	.LBB28_14
	movq	32(%rsp), %rax
	lock		decq	-8(%rax)
	jne	.LBB28_14
	movq	32(%rsp), %rdi
	addq	$-8, %rdi
	#MEMBARRIER
	callq	KGEN_CompilerRT_AlignedFree@PLT
.LBB28_14:
	callq	KGEN_CompilerRT_TracyIsEnabled@PLT
	testq	%rax, %rax
	je	.LBB28_15
	movq	48(%rsp), %rdi
	movq	40(%rsp), %rsi
	xorl	%edx, %edx
	callq	KGEN_CompilerRT_TracyZoneBegin@PLT
	movq	%rax, %rbx
	jmp	.LBB28_17
.LBB28_15:
	xorl	%ebx, %ebx
.LBB28_17:
	movq	%r12, %r11
	movq	112(%r12), %rcx
	movq	120(%r12), %rax
	movq	(%rcx), %r14
	movq	128(%r12), %rcx
	movq	(%rcx), %rcx
	xorl	%r15d, %r15d
	cmpq	%rcx, %rax
	setl	%r15b
	addq	%r14, %r15
	testq	%r15, %r15
	jle	.LBB28_20
	cmpq	%rcx, %rax
	cmovlq	%rax, %rcx
	imulq	%rax, %r14
	addq	%rcx, %r14
	.p2align	4
.LBB28_19:
	movq	136(%r11), %rsi
	movq	144(%r11), %rdx
	movq	152(%r11), %rcx
	movq	160(%r11), %r8
	movq	168(%r11), %r9
	movq	176(%r11), %rax
	movq	%rax, (%rsp)
	movq	%r14, %rdi
	callq	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0])_process_i_tile(::Int),transpose_b=0,dtype=f64"@PLT
	movq	%r12, %r11
	incq	%r14
	decq	%r15
	jne	.LBB28_19
.LBB28_20:
	testq	%rbx, %rbx
	je	.LBB28_22
	movq	%rbx, %rdi
	callq	KGEN_CompilerRT_TracyZoneEnd@PLT
	movq	%r12, %r11
.LBB28_22:
	movl	20(%rsp), %eax
	movl	%eax, 28(%rsp)
	vldmxcsr	28(%rsp)
	movq	24(%r11), %rdi
	addq	$56, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	jmpq	*16(%r11)
.Lfunc_end28:
	.size	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume", .Lfunc_end28-"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_async_closure_2_resume"
	.cfi_endproc

	.p2align	4
	.type	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3",@function
"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3":
	.cfi_startproc
	jmpq	*8(%rdi)
.Lfunc_end29:
	.size	"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3", .Lfunc_end29-"gemm::matmul_nr_blocked[::DType,::Bool](matrix::Matrix[$0]&,matrix::Matrix[$0],matrix::Matrix[$0]),dtype=f64,transpose_b=0_closure_3"
	.cfi_endproc

	.p2align	4
	.type	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]",@function
"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]":
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$4168, %rsp
	.cfi_def_cfa_offset 4224
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, %rbx
	movq	%rdi, 16(%rsp)
	movabsq	$-9223372036854775808, %rcx
	cmpq	$2, %rbx
	movl	$1, %eax
	cmovgeq	%rbx, %rax
	movq	%rax, 8(%rsp)
	testq	%rbx, %rbx
	je	.LBB30_1
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	16(%rdx), %rdx
	testq	%rdx, %rdx
	js	.LBB30_3
	movq	8(%r14), %rsi
	cmpq	$2, %rbx
	jge	.LBB30_6
	jmp	.LBB30_11
.LBB30_1:
	jmp	.LBB30_50
.LBB30_3:
	movq	%rdx, %rsi
	shrq	$56, %rsi
	andl	$31, %esi
	cmpq	$2, %rbx
	jl	.LBB30_11
.LBB30_6:
	movq	8(%rsp), %rax
	decq	%rax
	leaq	40(%r14), %rdi
	jmp	.LBB30_7
	.p2align	4
.LBB30_9:
	movq	-8(%rdi), %r8
.LBB30_10:
	addq	%r15, %rsi
	addq	%r8, %rsi
	addq	$24, %rdi
	decq	%rax
	je	.LBB30_11
.LBB30_7:
	movq	(%rdi), %r8
	testq	%r8, %r8
	jns	.LBB30_9
	shrq	$56, %r8
	andl	$31, %r8d
	jmp	.LBB30_10
.LBB30_11:
	cmpq	$23, %rsi
	jg	.LBB30_13
	movq	%rcx, 40(%rsp)
	jmp	.LBB30_14
.LBB30_13:
	addq	$7, %rsi
	movq	%rsi, %r12
	sarq	$3, %r12
	andq	$-8, %rsi
	addq	$8, %rsi
	movl	$1, %edi
	callq	KGEN_CompilerRT_AlignedAlloc@PLT
	movq	$1, (%rax)
	addq	$8, %rax
	movq	%rax, 24(%rsp)
	movq	$0, 32(%rsp)
	movabsq	$4611686018427387904, %rax
	orq	%r12, %rax
	movq	%rax, 40(%rsp)
	movq	16(%r14), %rdx
	testq	%r12, %r12
	js	.LBB30_14
	movq	$0, 4152(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 4160(%rsp)
	testq	%rdx, %rdx
	movq	%r15, 48(%rsp)
	js	.LBB30_40
	movq	(%r14), %rsi
	movq	8(%r14), %rdx
	leaq	56(%rsp), %rdi
	callq	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"@PLT
	cmpq	$2, %rbx
	jge	.LBB30_43
	jmp	.LBB30_48
.LBB30_14:
	testq	%rdx, %rdx
	js	.LBB30_15
	movq	(%r14), %rsi
	movq	8(%r14), %rdx
	jmp	.LBB30_17
.LBB30_15:
	shrq	$56, %rdx
	andl	$31, %edx
	movq	%r14, %rsi
.LBB30_17:
	leaq	24(%rsp), %r13
	movq	%r13, %rdi
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
	cmpq	$2, %rbx
	jl	.LBB30_49
	movabsq	$4611686018427387904, %r12
	leaq	56(%rsp), %rbp
	addq	$24, %r14
	decq	%rbx
	jmp	.LBB30_19
	.p2align	4
.LBB30_37:
	movq	4160(%rsp), %rdi
	movq	%rbp, %rsi
.LBB30_38:
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
	addq	$24, %r14
	decq	%rbx
	je	.LBB30_49
.LBB30_19:
	movq	32(%rsp), %rax
	movq	40(%rsp), %rcx
	movq	%rcx, %rsi
	shrq	$56, %rsi
	andl	$31, %esi
	testq	%rcx, %rcx
	cmovnsq	%rax, %rsi
	movq	16(%r14), %rdx
	testq	%rdx, %rdx
	js	.LBB30_20
	movq	8(%r14), %rdx
	addq	%r15, %rsi
	addq	%rdx, %rsi
	cmpq	$23, %rsi
	jg	.LBB30_27
.LBB30_23:
	movq	%r13, %rdi
	movq	16(%rsp), %rsi
	movq	%r15, %rdx
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
	movq	16(%r14), %rdx
	testq	%rdx, %rdx
	js	.LBB30_24
	movq	(%r14), %rsi
	movq	8(%r14), %rdx
	movq	%r13, %rdi
	jmp	.LBB30_38
	.p2align	4
.LBB30_20:
	shrq	$56, %rdx
	andl	$31, %edx
	addq	%r15, %rsi
	addq	%rdx, %rsi
	cmpq	$23, %rsi
	jle	.LBB30_23
.LBB30_27:
	testq	%rcx, %rcx
	js	.LBB30_29
	leaq	(,%rcx,8), %rdx
	cmpq	%r12, %rcx
	cmovbq	%rax, %rdx
	cmpq	%rdx, %rsi
	jle	.LBB30_30
.LBB30_29:
	movq	%r13, %rdi
	callq	"std::collections::string::string::String::_realloc_mutable(::String&,::Int)"@PLT
.LBB30_30:
	movq	$0, 4152(%rsp)
	movq	%r13, 4160(%rsp)
	movq	%rbp, %rdi
	movq	16(%rsp), %rsi
	movq	%r15, %rdx
	callq	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"@PLT
	movq	4152(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB30_32
	movq	4160(%rsp), %rdi
	movq	%rbp, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
	movq	$0, 4152(%rsp)
.LBB30_32:
	movq	16(%r14), %rdx
	testq	%rdx, %rdx
	js	.LBB30_33
	movq	(%r14), %rsi
	movq	8(%r14), %rdx
	jmp	.LBB30_35
.LBB30_24:
	shrq	$56, %rdx
	andl	$31, %edx
	movq	%r14, %rsi
	movq	%r13, %rdi
	jmp	.LBB30_38
.LBB30_33:
	shrq	$56, %rdx
	andl	$31, %edx
	movq	%r14, %rsi
.LBB30_35:
	movq	%rbp, %rdi
	callq	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"@PLT
	movq	4152(%rsp), %rdx
	cmpq	$4097, %rdx
	jl	.LBB30_37
	movq	4160(%rsp), %rdi
	movq	%rbp, %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
	movq	$0, 4152(%rsp)
	xorl	%edx, %edx
	jmp	.LBB30_37
.LBB30_40:
	shrq	$56, %rdx
	andl	$31, %edx
	movq	%r14, %rsi
	leaq	56(%rsp), %rdi
	callq	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"@PLT
	cmpq	$2, %rbx
	jl	.LBB30_48
.LBB30_43:
	addq	$24, %r14
	decq	8(%rsp)
	xorl	%r12d, %r12d
	leaq	56(%rsp), %rdi
	jmp	.LBB30_44
	.p2align	4
.LBB30_46:
	movq	(%rbp), %rbp
	movq	8(%r14,%r13,8), %rdx
.LBB30_47:
	movq	%r15, %rdi
	movq	%rbp, %rsi
	callq	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"@PLT
	movq	%r15, %rdi
	addq	$24, %r14
	cmpq	%r12, 8(%rsp)
	je	.LBB30_48
.LBB30_44:
	incq	%r12
	movq	%r12, %rax
	sarq	$63, %rax
	andq	%rbx, %rax
	leaq	(%rax,%rax,2), %r13
	leaq	(%r14,%r13,8), %rbp
	movq	%rdi, %r15
	movq	16(%rsp), %rsi
	movq	48(%rsp), %rdx
	callq	"std::format::_utils::_WriteBufferStack::write_string[::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::_WriteBufferStack[$0, $1, $2, $3]&,::StringSlice[$4, $5, $6]),W=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>],stack_buffer_bytes=4096,string.mut`2x1=0"@PLT
	movq	16(%r14,%r13,8), %rdx
	testq	%rdx, %rdx
	jns	.LBB30_46
	shrq	$56, %rdx
	andl	$31, %edx
	jmp	.LBB30_47
.LBB30_48:
	movq	4152(%rsp), %rdx
	movq	4160(%rsp), %rdi
	leaq	56(%rsp), %rsi
	callq	"std::collections::string::string::String::_iadd[LITImmutOrigin,::Origin[::Bool(False), $0]](::String&,::Span[::Bool(False), $0, ::SIMD[::DType(uint8), ::Int(1)], $1])"@PLT
.LBB30_49:
	movq	24(%rsp), %rax
	movq	32(%rsp), %rdx
	movq	40(%rsp), %rcx
.LBB30_50:
	addq	$4168, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end30:
	.size	"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]", .Lfunc_end30-"std::collections::string::string_slice::StringSlice::join[::Copyable & ::Writable,::Bool,LITOrigin[$4._mlir_value],::Origin[$4, $5]](::StringSlice[$0, $1, $2],::Span[$4, $5, $3, $6])_REMOVED_ARG,mut=0,T=[typevalue<#kgen.instref<\"std::collections::string::string::String\">>, struct<(pointer<none>, index, index) memoryOnly>]"
	.cfi_endproc

	.type	static_string_c44bdff4074eecdb,@object
	.section	.rodata,"a",@progbits
	.p2align	4, 0x0
static_string_c44bdff4074eecdb:
	.zero	1
	.size	static_string_c44bdff4074eecdb, 1

	.type	static_string_a61c3395ab9379d9,@object
	.p2align	4, 0x0
static_string_a61c3395ab9379d9:
	.asciz	"Runtime"
	.size	static_string_a61c3395ab9379d9, 8

	.type	static_string_2b3f504061b33816,@object
	.p2align	4, 0x0
static_string_2b3f504061b33816:
	.asciz	"task"
	.size	static_string_2b3f504061b33816, 5

	.type	static_string_0c475e2a8e1ec05d,@object
	.p2align	4, 0x0
static_string_0c475e2a8e1ec05d:
	.asciz	"Tracy"
	.size	static_string_0c475e2a8e1ec05d, 6

	.type	static_string_44fd141e40b306d5,@object
	.p2align	4, 0x0
static_string_44fd141e40b306d5:
	.asciz	", "
	.size	static_string_44fd141e40b306d5, 3

	.type	static_string_f9c5d72f244f07d1,@object
	.p2align	4, 0x0
static_string_f9c5d72f244f07d1:
	.asciz	"`Optional.value()` called on empty `Optional`. Consider using `if optional:` to check whether the `Optional` is empty before calling `.value()`, or use `.or_else()` to provide a default value."
	.size	static_string_f9c5d72f244f07d1, 193

	.section	".note.GNU-stack","",@progbits
