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
	movss	(%rdi,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	(%rsi,%rcx,4), %xmm0
	movss	%xmm0, (%rdx,%rcx,4)
	movss	4(%rdi,%rcx,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	addss	4(%rsi,%rcx,4), %xmm0
	movss	%xmm0, 4(%rdx,%rcx,4)
	movss	8(%rdi,%rcx,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	addss	8(%rsi,%rcx,4), %xmm0
	movss	%xmm0, 8(%rdx,%rcx,4)
	movss	12(%rdi,%rcx,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	addss	12(%rsi,%rcx,4), %xmm0
	movss	%xmm0, 12(%rdx,%rcx,4)
	addq	$4, %rcx
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
