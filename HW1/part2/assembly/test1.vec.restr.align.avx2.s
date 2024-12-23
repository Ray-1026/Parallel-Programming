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
	vmovaps	(%rdi,%rcx,4), %ymm0
	vmovaps	32(%rdi,%rcx,4), %ymm1
	vmovaps	64(%rdi,%rcx,4), %ymm2
	vmovaps	96(%rdi,%rcx,4), %ymm3
	vaddps	(%rsi,%rcx,4), %ymm0, %ymm0
	vaddps	32(%rsi,%rcx,4), %ymm1, %ymm1
	vaddps	64(%rsi,%rcx,4), %ymm2, %ymm2
	vaddps	96(%rsi,%rcx,4), %ymm3, %ymm3
	vmovaps	%ymm0, (%rdx,%rcx,4)
	vmovaps	%ymm1, 32(%rdx,%rcx,4)
	vmovaps	%ymm2, 64(%rdx,%rcx,4)
	vmovaps	%ymm3, 96(%rdx,%rcx,4)
	addq	$32, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	addl	$1, %eax
	cmpl	$20000000, %eax                 # imm = 0x1312D00
	jne	.LBB0_1
# %bb.4:
	vzeroupper
	retq
.Lfunc_end0:
	.size	test1, .Lfunc_end0-test1
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 11.1.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
