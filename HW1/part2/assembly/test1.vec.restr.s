	.text
	.file	"test1.c"
	.globl	test1                           # -- Begin function test1
	.p2align	4, 0x90
	.type	test1,@function
test1:                                  # @test1
	.cfi_startproc
# %bb.0:
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movups	(%rdi,%rcx,4), %xmm0
	movups	16(%rdi,%rcx,4), %xmm1
	movups	(%rsi,%rcx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%rsi,%rcx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%rdx,%rcx,4)
	movups	%xmm0, 16(%rdx,%rcx,4)
	movups	32(%rdi,%rcx,4), %xmm0
	movups	48(%rdi,%rcx,4), %xmm1
	movups	32(%rsi,%rcx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	48(%rsi,%rcx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, 32(%rdx,%rcx,4)
	movups	%xmm0, 48(%rdx,%rcx,4)
	addq	$16, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	addl	$1, %eax
	cmpl	$20000000, %eax                 # imm = 0x1312D00
	jne	.LBB0_1
# %bb.4:
	retq
.Lfunc_end0:
	.size	test1, .Lfunc_end0-test1
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 11.1.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
