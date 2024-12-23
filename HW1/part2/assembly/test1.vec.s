	.text
	.file	"test1.c"
	.globl	test1                           # -- Begin function test1
	.p2align	4, 0x90
	.type	test1,@function
test1:                                  # @test1
	.cfi_startproc
# %bb.0:
	leaq	4096(%rdx), %rax
	leaq	4096(%rdi), %rcx
	cmpq	%rdx, %rcx
	seta	%r9b
	leaq	4096(%rsi), %r8
	cmpq	%rdi, %rax
	seta	%cl
	andb	%r9b, %cl
	cmpq	%rdx, %r8
	seta	%r9b
	cmpq	%rsi, %rax
	seta	%r8b
	andb	%r9b, %r8b
	orb	%cl, %r8b
	xorl	%ecx, %ecx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_3:                                #   in Loop: Header=BB0_1 Depth=1
	addl	$1, %ecx
	cmpl	$20000000, %ecx                 # imm = 0x1312D00
	je	.LBB0_4
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
                                        #     Child Loop BB0_5 Depth 2
	xorl	%eax, %eax
	testb	%r8b, %r8b
	je	.LBB0_2
	.p2align	4, 0x90
.LBB0_5:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%rdi,%rax,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	(%rsi,%rax,4), %xmm0
	movss	%xmm0, (%rdx,%rax,4)
	movss	4(%rdi,%rax,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	addss	4(%rsi,%rax,4), %xmm0
	movss	%xmm0, 4(%rdx,%rax,4)
	movss	8(%rdi,%rax,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	addss	8(%rsi,%rax,4), %xmm0
	movss	%xmm0, 8(%rdx,%rax,4)
	movss	12(%rdi,%rax,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	addss	12(%rsi,%rax,4), %xmm0
	movss	%xmm0, 12(%rdx,%rax,4)
	addq	$4, %rax
	cmpq	$1024, %rax                     # imm = 0x400
	jne	.LBB0_5
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movups	(%rdi,%rax,4), %xmm0
	movups	16(%rdi,%rax,4), %xmm1
	movups	(%rsi,%rax,4), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%rsi,%rax,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%rdx,%rax,4)
	movups	%xmm0, 16(%rdx,%rax,4)
	movups	32(%rdi,%rax,4), %xmm0
	movups	48(%rdi,%rax,4), %xmm1
	movups	32(%rsi,%rax,4), %xmm2
	addps	%xmm0, %xmm2
	movups	48(%rsi,%rax,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, 32(%rdx,%rax,4)
	movups	%xmm0, 48(%rdx,%rax,4)
	addq	$16, %rax
	cmpq	$1024, %rax                     # imm = 0x400
	jne	.LBB0_2
	jmp	.LBB0_3
.LBB0_4:
	retq
.Lfunc_end0:
	.size	test1, .Lfunc_end0-test1
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 11.1.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
