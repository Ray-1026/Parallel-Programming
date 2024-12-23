	.text
	.file	"test2.c"
	.globl	test2                           # -- Begin function test2
	.p2align	4, 0x90
	.type	test2,@function
test2:                                  # @test2
	.cfi_startproc
# %bb.0:
	xorl	%r8d, %r8d
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_7:                                #   in Loop: Header=BB0_1 Depth=1
	addl	$1, %r8d
	cmpl	$20000000, %r8d                 # imm = 0x1312D00
	je	.LBB0_8
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	xorl	%ecx, %ecx
	jmp	.LBB0_2
	.p2align	4, 0x90
.LBB0_6:                                #   in Loop: Header=BB0_2 Depth=2
	addq	$2, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	je	.LBB0_7
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	(%rdi,%rcx,4), %eax
	movl	%eax, (%rdx,%rcx,4)
	movss	(%rsi,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movd	%eax, %xmm1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB0_4
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=2
	movss	%xmm0, (%rdx,%rcx,4)
.LBB0_4:                                #   in Loop: Header=BB0_2 Depth=2
	movl	4(%rdi,%rcx,4), %eax
	movl	%eax, 4(%rdx,%rcx,4)
	movss	4(%rsi,%rcx,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	movd	%eax, %xmm1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB0_6
# %bb.5:                                #   in Loop: Header=BB0_2 Depth=2
	movss	%xmm0, 4(%rdx,%rcx,4)
	jmp	.LBB0_6
.LBB0_8:
	retq
.Lfunc_end0:
	.size	test2, .Lfunc_end0-test2
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 11.1.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
