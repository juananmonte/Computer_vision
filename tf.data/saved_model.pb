¦ū-
åµ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ś
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ń8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ŅŃ%
¬
$separable_conv2d_12/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_12/depthwise_kernel
„
8separable_conv2d_12/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_12/depthwise_kernel*&
_output_shapes
:*
dtype0
¬
$separable_conv2d_12/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$separable_conv2d_12/pointwise_kernel
„
8separable_conv2d_12/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_12/pointwise_kernel*&
_output_shapes
: *
dtype0

separable_conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameseparable_conv2d_12/bias

,separable_conv2d_12/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_12/bias*
_output_shapes
: *
dtype0

batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_14/gamma

0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes
: *
dtype0

batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_14/beta

/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes
: *
dtype0

"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_14/moving_mean

6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes
: *
dtype0
¤
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_14/moving_variance

:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes
: *
dtype0
¬
$separable_conv2d_13/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$separable_conv2d_13/depthwise_kernel
„
8separable_conv2d_13/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_13/depthwise_kernel*&
_output_shapes
: *
dtype0
¬
$separable_conv2d_13/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*5
shared_name&$separable_conv2d_13/pointwise_kernel
„
8separable_conv2d_13/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_13/pointwise_kernel*&
_output_shapes
: @*
dtype0

separable_conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameseparable_conv2d_13/bias

,separable_conv2d_13/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_13/bias*
_output_shapes
:@*
dtype0

batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_15/gamma

0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:@*
dtype0

batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_15/beta

/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:@*
dtype0

"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_15/moving_mean

6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_15/moving_variance

:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:@*
dtype0
¬
$separable_conv2d_14/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$separable_conv2d_14/depthwise_kernel
„
8separable_conv2d_14/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_14/depthwise_kernel*&
_output_shapes
:@*
dtype0
¬
$separable_conv2d_14/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*5
shared_name&$separable_conv2d_14/pointwise_kernel
„
8separable_conv2d_14/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_14/pointwise_kernel*&
_output_shapes
:@@*
dtype0

separable_conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameseparable_conv2d_14/bias

,separable_conv2d_14/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_14/bias*
_output_shapes
:@*
dtype0

batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_16/gamma

0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:@*
dtype0

batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_16/beta

/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:@*
dtype0

"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_16/moving_mean

6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_16/moving_variance

:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:@*
dtype0
¬
$separable_conv2d_15/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$separable_conv2d_15/depthwise_kernel
„
8separable_conv2d_15/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_15/depthwise_kernel*&
_output_shapes
:@*
dtype0
­
$separable_conv2d_15/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$separable_conv2d_15/pointwise_kernel
¦
8separable_conv2d_15/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_15/pointwise_kernel*'
_output_shapes
:@*
dtype0

separable_conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameseparable_conv2d_15/bias

,separable_conv2d_15/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_15/bias*
_output_shapes	
:*
dtype0

batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_17/gamma

0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes	
:*
dtype0

batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_17/beta

/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes	
:*
dtype0

"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_17/moving_mean

6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_17/moving_variance

:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes	
:*
dtype0
­
$separable_conv2d_16/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_16/depthwise_kernel
¦
8separable_conv2d_16/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_16/depthwise_kernel*'
_output_shapes
:*
dtype0
®
$separable_conv2d_16/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_16/pointwise_kernel
§
8separable_conv2d_16/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_16/pointwise_kernel*(
_output_shapes
:*
dtype0

separable_conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameseparable_conv2d_16/bias

,separable_conv2d_16/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_16/bias*
_output_shapes	
:*
dtype0

batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_18/gamma

0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes	
:*
dtype0

batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_18/beta

/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes	
:*
dtype0

"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_18/moving_mean

6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_18/moving_variance

:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:*
dtype0
­
$separable_conv2d_17/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_17/depthwise_kernel
¦
8separable_conv2d_17/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_17/depthwise_kernel*'
_output_shapes
:*
dtype0
®
$separable_conv2d_17/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_17/pointwise_kernel
§
8separable_conv2d_17/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_17/pointwise_kernel*(
_output_shapes
:*
dtype0

separable_conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameseparable_conv2d_17/bias

,separable_conv2d_17/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_17/bias*
_output_shapes	
:*
dtype0

batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_19/gamma

0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes	
:*
dtype0

batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_19/beta

/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes	
:*
dtype0

"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_19/moving_mean

6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_19/moving_variance

:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes	
:*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
$*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
$*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0

batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_20/gamma

0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes	
:*
dtype0

batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_20/beta

/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes	
:*
dtype0

"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_20/moving_mean

6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes	
:*
dtype0
„
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_20/moving_variance

:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes	
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Ō
8Adagrad/separable_conv2d_12/depthwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adagrad/separable_conv2d_12/depthwise_kernel/accumulator
Ķ
LAdagrad/separable_conv2d_12/depthwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_12/depthwise_kernel/accumulator*&
_output_shapes
:*
dtype0
Ō
8Adagrad/separable_conv2d_12/pointwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adagrad/separable_conv2d_12/pointwise_kernel/accumulator
Ķ
LAdagrad/separable_conv2d_12/pointwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_12/pointwise_kernel/accumulator*&
_output_shapes
: *
dtype0
°
,Adagrad/separable_conv2d_12/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adagrad/separable_conv2d_12/bias/accumulator
©
@Adagrad/separable_conv2d_12/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/separable_conv2d_12/bias/accumulator*
_output_shapes
: *
dtype0
ø
0Adagrad/batch_normalization_14/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adagrad/batch_normalization_14/gamma/accumulator
±
DAdagrad/batch_normalization_14/gamma/accumulator/Read/ReadVariableOpReadVariableOp0Adagrad/batch_normalization_14/gamma/accumulator*
_output_shapes
: *
dtype0
¶
/Adagrad/batch_normalization_14/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/Adagrad/batch_normalization_14/beta/accumulator
Æ
CAdagrad/batch_normalization_14/beta/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_14/beta/accumulator*
_output_shapes
: *
dtype0
Ō
8Adagrad/separable_conv2d_13/depthwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adagrad/separable_conv2d_13/depthwise_kernel/accumulator
Ķ
LAdagrad/separable_conv2d_13/depthwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_13/depthwise_kernel/accumulator*&
_output_shapes
: *
dtype0
Ō
8Adagrad/separable_conv2d_13/pointwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*I
shared_name:8Adagrad/separable_conv2d_13/pointwise_kernel/accumulator
Ķ
LAdagrad/separable_conv2d_13/pointwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_13/pointwise_kernel/accumulator*&
_output_shapes
: @*
dtype0
°
,Adagrad/separable_conv2d_13/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adagrad/separable_conv2d_13/bias/accumulator
©
@Adagrad/separable_conv2d_13/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/separable_conv2d_13/bias/accumulator*
_output_shapes
:@*
dtype0
ø
0Adagrad/batch_normalization_15/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adagrad/batch_normalization_15/gamma/accumulator
±
DAdagrad/batch_normalization_15/gamma/accumulator/Read/ReadVariableOpReadVariableOp0Adagrad/batch_normalization_15/gamma/accumulator*
_output_shapes
:@*
dtype0
¶
/Adagrad/batch_normalization_15/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adagrad/batch_normalization_15/beta/accumulator
Æ
CAdagrad/batch_normalization_15/beta/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_15/beta/accumulator*
_output_shapes
:@*
dtype0
Ō
8Adagrad/separable_conv2d_14/depthwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adagrad/separable_conv2d_14/depthwise_kernel/accumulator
Ķ
LAdagrad/separable_conv2d_14/depthwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_14/depthwise_kernel/accumulator*&
_output_shapes
:@*
dtype0
Ō
8Adagrad/separable_conv2d_14/pointwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*I
shared_name:8Adagrad/separable_conv2d_14/pointwise_kernel/accumulator
Ķ
LAdagrad/separable_conv2d_14/pointwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_14/pointwise_kernel/accumulator*&
_output_shapes
:@@*
dtype0
°
,Adagrad/separable_conv2d_14/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adagrad/separable_conv2d_14/bias/accumulator
©
@Adagrad/separable_conv2d_14/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/separable_conv2d_14/bias/accumulator*
_output_shapes
:@*
dtype0
ø
0Adagrad/batch_normalization_16/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adagrad/batch_normalization_16/gamma/accumulator
±
DAdagrad/batch_normalization_16/gamma/accumulator/Read/ReadVariableOpReadVariableOp0Adagrad/batch_normalization_16/gamma/accumulator*
_output_shapes
:@*
dtype0
¶
/Adagrad/batch_normalization_16/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adagrad/batch_normalization_16/beta/accumulator
Æ
CAdagrad/batch_normalization_16/beta/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_16/beta/accumulator*
_output_shapes
:@*
dtype0
Ō
8Adagrad/separable_conv2d_15/depthwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adagrad/separable_conv2d_15/depthwise_kernel/accumulator
Ķ
LAdagrad/separable_conv2d_15/depthwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_15/depthwise_kernel/accumulator*&
_output_shapes
:@*
dtype0
Õ
8Adagrad/separable_conv2d_15/pointwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adagrad/separable_conv2d_15/pointwise_kernel/accumulator
Ī
LAdagrad/separable_conv2d_15/pointwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_15/pointwise_kernel/accumulator*'
_output_shapes
:@*
dtype0
±
,Adagrad/separable_conv2d_15/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adagrad/separable_conv2d_15/bias/accumulator
Ŗ
@Adagrad/separable_conv2d_15/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/separable_conv2d_15/bias/accumulator*
_output_shapes	
:*
dtype0
¹
0Adagrad/batch_normalization_17/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adagrad/batch_normalization_17/gamma/accumulator
²
DAdagrad/batch_normalization_17/gamma/accumulator/Read/ReadVariableOpReadVariableOp0Adagrad/batch_normalization_17/gamma/accumulator*
_output_shapes	
:*
dtype0
·
/Adagrad/batch_normalization_17/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adagrad/batch_normalization_17/beta/accumulator
°
CAdagrad/batch_normalization_17/beta/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_17/beta/accumulator*
_output_shapes	
:*
dtype0
Õ
8Adagrad/separable_conv2d_16/depthwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adagrad/separable_conv2d_16/depthwise_kernel/accumulator
Ī
LAdagrad/separable_conv2d_16/depthwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_16/depthwise_kernel/accumulator*'
_output_shapes
:*
dtype0
Ö
8Adagrad/separable_conv2d_16/pointwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adagrad/separable_conv2d_16/pointwise_kernel/accumulator
Ļ
LAdagrad/separable_conv2d_16/pointwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_16/pointwise_kernel/accumulator*(
_output_shapes
:*
dtype0
±
,Adagrad/separable_conv2d_16/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adagrad/separable_conv2d_16/bias/accumulator
Ŗ
@Adagrad/separable_conv2d_16/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/separable_conv2d_16/bias/accumulator*
_output_shapes	
:*
dtype0
¹
0Adagrad/batch_normalization_18/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adagrad/batch_normalization_18/gamma/accumulator
²
DAdagrad/batch_normalization_18/gamma/accumulator/Read/ReadVariableOpReadVariableOp0Adagrad/batch_normalization_18/gamma/accumulator*
_output_shapes	
:*
dtype0
·
/Adagrad/batch_normalization_18/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adagrad/batch_normalization_18/beta/accumulator
°
CAdagrad/batch_normalization_18/beta/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_18/beta/accumulator*
_output_shapes	
:*
dtype0
Õ
8Adagrad/separable_conv2d_17/depthwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adagrad/separable_conv2d_17/depthwise_kernel/accumulator
Ī
LAdagrad/separable_conv2d_17/depthwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_17/depthwise_kernel/accumulator*'
_output_shapes
:*
dtype0
Ö
8Adagrad/separable_conv2d_17/pointwise_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adagrad/separable_conv2d_17/pointwise_kernel/accumulator
Ļ
LAdagrad/separable_conv2d_17/pointwise_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/separable_conv2d_17/pointwise_kernel/accumulator*(
_output_shapes
:*
dtype0
±
,Adagrad/separable_conv2d_17/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adagrad/separable_conv2d_17/bias/accumulator
Ŗ
@Adagrad/separable_conv2d_17/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/separable_conv2d_17/bias/accumulator*
_output_shapes	
:*
dtype0
¹
0Adagrad/batch_normalization_19/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adagrad/batch_normalization_19/gamma/accumulator
²
DAdagrad/batch_normalization_19/gamma/accumulator/Read/ReadVariableOpReadVariableOp0Adagrad/batch_normalization_19/gamma/accumulator*
_output_shapes	
:*
dtype0
·
/Adagrad/batch_normalization_19/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adagrad/batch_normalization_19/beta/accumulator
°
CAdagrad/batch_normalization_19/beta/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_19/beta/accumulator*
_output_shapes	
:*
dtype0
¢
"Adagrad/dense_4/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
$*3
shared_name$"Adagrad/dense_4/kernel/accumulator

6Adagrad/dense_4/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_4/kernel/accumulator* 
_output_shapes
:
$*
dtype0

 Adagrad/dense_4/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adagrad/dense_4/bias/accumulator

4Adagrad/dense_4/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_4/bias/accumulator*
_output_shapes	
:*
dtype0
¹
0Adagrad/batch_normalization_20/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adagrad/batch_normalization_20/gamma/accumulator
²
DAdagrad/batch_normalization_20/gamma/accumulator/Read/ReadVariableOpReadVariableOp0Adagrad/batch_normalization_20/gamma/accumulator*
_output_shapes	
:*
dtype0
·
/Adagrad/batch_normalization_20/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adagrad/batch_normalization_20/beta/accumulator
°
CAdagrad/batch_normalization_20/beta/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_20/beta/accumulator*
_output_shapes	
:*
dtype0
”
"Adagrad/dense_5/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adagrad/dense_5/kernel/accumulator

6Adagrad/dense_5/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_5/kernel/accumulator*
_output_shapes
:	*
dtype0

 Adagrad/dense_5/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adagrad/dense_5/bias/accumulator

4Adagrad/dense_5/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_5/bias/accumulator*
_output_shapes
:*
dtype0

NoOpNoOp
÷Č
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*±Č
value¦ČB¢Č BČ

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer-26
layer_with_weights-13
layer-27
layer-28
layer_with_weights-14
layer-29
	optimizer
 	variables
!regularization_losses
"trainable_variables
#	keras_api
$
signatures

%depthwise_kernel
&pointwise_kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api

0axis
	1gamma
2beta
3moving_mean
4moving_variance
5	variables
6regularization_losses
7trainable_variables
8	keras_api
R
9	variables
:regularization_losses
;trainable_variables
<	keras_api
R
=	variables
>regularization_losses
?trainable_variables
@	keras_api

Adepthwise_kernel
Bpointwise_kernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
R
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api

Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api

Udepthwise_kernel
Vpointwise_kernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
R
\	variables
]regularization_losses
^trainable_variables
_	keras_api

`axis
	agamma
bbeta
cmoving_mean
dmoving_variance
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
R
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
R
m	variables
nregularization_losses
otrainable_variables
p	keras_api

qdepthwise_kernel
rpointwise_kernel
sbias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
R
x	variables
yregularization_losses
ztrainable_variables
{	keras_api

|axis
	}gamma
~beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api

depthwise_kernel
pointwise_kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api

depthwise_kernel
pointwise_kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
V
 	variables
”regularization_losses
¢trainable_variables
£	keras_api
 
	¤axis

„gamma
	¦beta
§moving_mean
Ømoving_variance
©	variables
Ŗregularization_losses
«trainable_variables
¬	keras_api
V
­	variables
®regularization_losses
Ætrainable_variables
°	keras_api
V
±	variables
²regularization_losses
³trainable_variables
“	keras_api
V
µ	variables
¶regularization_losses
·trainable_variables
ø	keras_api
n
¹kernel
	ŗbias
»	variables
¼regularization_losses
½trainable_variables
¾	keras_api
V
æ	variables
Ąregularization_losses
Įtrainable_variables
Ā	keras_api
 
	Ćaxis

Ägamma
	Åbeta
Ęmoving_mean
Ēmoving_variance
Č	variables
Éregularization_losses
Źtrainable_variables
Ė	keras_api
V
Ģ	variables
Ķregularization_losses
Ītrainable_variables
Ļ	keras_api
n
Škernel
	Ńbias
Ņ	variables
Óregularization_losses
Ōtrainable_variables
Õ	keras_api

	Öiter

×decay
Ųlearning_rate%accumulator’&accumulator'accumulator1accumulator2accumulatorAaccumulatorBaccumulatorCaccumulatorMaccumulatorNaccumulatorUaccumulatorVaccumulatorWaccumulatoraaccumulatorbaccumulatorqaccumulatorraccumulatorsaccumulator}accumulator~accumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulator„accumulator¦accumulator¹accumulatorŗaccumulatorÄaccumulatorÅaccumulator Šaccumulator”Ńaccumulator¢

%0
&1
'2
13
24
35
46
A7
B8
C9
M10
N11
O12
P13
U14
V15
W16
a17
b18
c19
d20
q21
r22
s23
}24
~25
26
27
28
29
30
31
32
33
34
35
36
37
„38
¦39
§40
Ø41
¹42
ŗ43
Ä44
Å45
Ę46
Ē47
Š48
Ń49
 
¦
%0
&1
'2
13
24
A5
B6
C7
M8
N9
U10
V11
W12
a13
b14
q15
r16
s17
}18
~19
20
21
22
23
24
25
26
27
„28
¦29
¹30
ŗ31
Ä32
Å33
Š34
Ń35
²
Łnon_trainable_variables
Ślayer_metrics
 	variables
 Ūlayer_regularization_losses
!regularization_losses
Ümetrics
Żlayers
"trainable_variables
 
zx
VARIABLE_VALUE$separable_conv2d_12/depthwise_kernel@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_12/pointwise_kernel@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_12/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
'2
 

%0
&1
'2
²
Žnon_trainable_variables
ßlayer_metrics
(	variables
 ąlayer_regularization_losses
)regularization_losses
įmetrics
ālayers
*trainable_variables
 
 
 
²
ćnon_trainable_variables
älayer_metrics
,	variables
 ålayer_regularization_losses
-regularization_losses
ęmetrics
ēlayers
.trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

10
21
32
43
 

10
21
²
čnon_trainable_variables
élayer_metrics
5	variables
 źlayer_regularization_losses
6regularization_losses
ėmetrics
ģlayers
7trainable_variables
 
 
 
²
ķnon_trainable_variables
īlayer_metrics
9	variables
 ļlayer_regularization_losses
:regularization_losses
šmetrics
ńlayers
;trainable_variables
 
 
 
²
ņnon_trainable_variables
ólayer_metrics
=	variables
 ōlayer_regularization_losses
>regularization_losses
õmetrics
ölayers
?trainable_variables
zx
VARIABLE_VALUE$separable_conv2d_13/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_13/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
C2
 

A0
B1
C2
²
÷non_trainable_variables
ųlayer_metrics
D	variables
 łlayer_regularization_losses
Eregularization_losses
śmetrics
ūlayers
Ftrainable_variables
 
 
 
²
ünon_trainable_variables
żlayer_metrics
H	variables
 žlayer_regularization_losses
Iregularization_losses
’metrics
layers
Jtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
O2
P3
 

M0
N1
²
non_trainable_variables
layer_metrics
Q	variables
 layer_regularization_losses
Rregularization_losses
metrics
layers
Strainable_variables
zx
VARIABLE_VALUE$separable_conv2d_14/depthwise_kernel@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_14/pointwise_kernel@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
W2
 

U0
V1
W2
²
non_trainable_variables
layer_metrics
X	variables
 layer_regularization_losses
Yregularization_losses
metrics
layers
Ztrainable_variables
 
 
 
²
non_trainable_variables
layer_metrics
\	variables
 layer_regularization_losses
]regularization_losses
metrics
layers
^trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

a0
b1
c2
d3
 

a0
b1
²
non_trainable_variables
layer_metrics
e	variables
 layer_regularization_losses
fregularization_losses
metrics
layers
gtrainable_variables
 
 
 
²
non_trainable_variables
layer_metrics
i	variables
 layer_regularization_losses
jregularization_losses
metrics
layers
ktrainable_variables
 
 
 
²
non_trainable_variables
layer_metrics
m	variables
 layer_regularization_losses
nregularization_losses
metrics
layers
otrainable_variables
zx
VARIABLE_VALUE$separable_conv2d_15/depthwise_kernel@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_15/pointwise_kernel@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_15/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
s2
 

q0
r1
s2
²
non_trainable_variables
 layer_metrics
t	variables
 ”layer_regularization_losses
uregularization_losses
¢metrics
£layers
vtrainable_variables
 
 
 
²
¤non_trainable_variables
„layer_metrics
x	variables
 ¦layer_regularization_losses
yregularization_losses
§metrics
Ølayers
ztrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

}0
~1
2
3
 

}0
~1
µ
©non_trainable_variables
Ŗlayer_metrics
	variables
 «layer_regularization_losses
regularization_losses
¬metrics
­layers
trainable_variables
zx
VARIABLE_VALUE$separable_conv2d_16/depthwise_kernel@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_16/pointwise_kernel@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEseparable_conv2d_16/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

0
1
2
µ
®non_trainable_variables
Ælayer_metrics
	variables
 °layer_regularization_losses
regularization_losses
±metrics
²layers
trainable_variables
 
 
 
µ
³non_trainable_variables
“layer_metrics
	variables
 µlayer_regularization_losses
regularization_losses
¶metrics
·layers
trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
µ
ønon_trainable_variables
¹layer_metrics
	variables
 ŗlayer_regularization_losses
regularization_losses
»metrics
¼layers
trainable_variables
{y
VARIABLE_VALUE$separable_conv2d_17/depthwise_kernelAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE$separable_conv2d_17/pointwise_kernelAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEseparable_conv2d_17/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

0
1
2
µ
½non_trainable_variables
¾layer_metrics
	variables
 ælayer_regularization_losses
regularization_losses
Ąmetrics
Įlayers
trainable_variables
 
 
 
µ
Ānon_trainable_variables
Ćlayer_metrics
 	variables
 Älayer_regularization_losses
”regularization_losses
Åmetrics
Ęlayers
¢trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_19/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_19/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_19/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_19/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
„0
¦1
§2
Ø3
 

„0
¦1
µ
Ēnon_trainable_variables
Člayer_metrics
©	variables
 Élayer_regularization_losses
Ŗregularization_losses
Źmetrics
Ėlayers
«trainable_variables
 
 
 
µ
Ģnon_trainable_variables
Ķlayer_metrics
­	variables
 Īlayer_regularization_losses
®regularization_losses
Ļmetrics
Šlayers
Ætrainable_variables
 
 
 
µ
Ńnon_trainable_variables
Ņlayer_metrics
±	variables
 Ólayer_regularization_losses
²regularization_losses
Ōmetrics
Õlayers
³trainable_variables
 
 
 
µ
Önon_trainable_variables
×layer_metrics
µ	variables
 Ųlayer_regularization_losses
¶regularization_losses
Łmetrics
Ślayers
·trainable_variables
[Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

¹0
ŗ1
 

¹0
ŗ1
µ
Ūnon_trainable_variables
Ülayer_metrics
»	variables
 Żlayer_regularization_losses
¼regularization_losses
Žmetrics
ßlayers
½trainable_variables
 
 
 
µ
ąnon_trainable_variables
įlayer_metrics
æ	variables
 ālayer_regularization_losses
Ąregularization_losses
ćmetrics
älayers
Įtrainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_20/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_20/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_20/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_20/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ä0
Å1
Ę2
Ē3
 

Ä0
Å1
µ
ånon_trainable_variables
ęlayer_metrics
Č	variables
 ēlayer_regularization_losses
Éregularization_losses
čmetrics
élayers
Źtrainable_variables
 
 
 
µ
źnon_trainable_variables
ėlayer_metrics
Ģ	variables
 ģlayer_regularization_losses
Ķregularization_losses
ķmetrics
īlayers
Ītrainable_variables
[Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

Š0
Ń1
 

Š0
Ń1
µ
ļnon_trainable_variables
šlayer_metrics
Ņ	variables
 ńlayer_regularization_losses
Óregularization_losses
ņmetrics
ólayers
Ōtrainable_variables
KI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
m
30
41
O2
P3
c4
d5
6
7
8
9
§10
Ø11
Ę12
Ē13
 
 

ō0
õ1
ę
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
 
 
 
 
 
 
 
 
 
 

30
41
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

O0
P1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

c0
d1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

§0
Ø1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Ę0
Ē1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ötotal

÷count
ų	variables
ł	keras_api
I

śtotal

ūcount
ü
_fn_kwargs
ż	variables
ž	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ö0
÷1

ų	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ś0
ū1

ż	variables
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_12/depthwise_kernel/accumulatorflayer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_12/pointwise_kernel/accumulatorflayer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adagrad/separable_conv2d_12/bias/accumulatorZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE0Adagrad/batch_normalization_14/gamma/accumulator[layer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE/Adagrad/batch_normalization_14/beta/accumulatorZlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_13/depthwise_kernel/accumulatorflayer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_13/pointwise_kernel/accumulatorflayer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adagrad/separable_conv2d_13/bias/accumulatorZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE0Adagrad/batch_normalization_15/gamma/accumulator[layer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE/Adagrad/batch_normalization_15/beta/accumulatorZlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_14/depthwise_kernel/accumulatorflayer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_14/pointwise_kernel/accumulatorflayer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adagrad/separable_conv2d_14/bias/accumulatorZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE0Adagrad/batch_normalization_16/gamma/accumulator[layer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE/Adagrad/batch_normalization_16/beta/accumulatorZlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_15/depthwise_kernel/accumulatorflayer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_15/pointwise_kernel/accumulatorflayer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adagrad/separable_conv2d_15/bias/accumulatorZlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE0Adagrad/batch_normalization_17/gamma/accumulator[layer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE/Adagrad/batch_normalization_17/beta/accumulatorZlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_16/depthwise_kernel/accumulatorflayer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
µ²
VARIABLE_VALUE8Adagrad/separable_conv2d_16/pointwise_kernel/accumulatorflayer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adagrad/separable_conv2d_16/bias/accumulatorZlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE0Adagrad/batch_normalization_18/gamma/accumulator[layer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE/Adagrad/batch_normalization_18/beta/accumulatorZlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
¶³
VARIABLE_VALUE8Adagrad/separable_conv2d_17/depthwise_kernel/accumulatorglayer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
¶³
VARIABLE_VALUE8Adagrad/separable_conv2d_17/pointwise_kernel/accumulatorglayer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adagrad/separable_conv2d_17/bias/accumulator[layer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE0Adagrad/batch_normalization_19/gamma/accumulator\layer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
”
VARIABLE_VALUE/Adagrad/batch_normalization_19/beta/accumulator[layer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adagrad/dense_4/kernel/accumulator]layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adagrad/dense_4/bias/accumulator[layer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE0Adagrad/batch_normalization_20/gamma/accumulator\layer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
”
VARIABLE_VALUE/Adagrad/batch_normalization_20/beta/accumulator[layer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adagrad/dense_5/kernel/accumulator]layer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adagrad/dense_5/bias/accumulator[layer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE

)serving_default_separable_conv2d_12_inputPlaceholder*/
_output_shapes
:’’’’’’’’’00*
dtype0*$
shape:’’’’’’’’’00
ü
StatefulPartitionedCallStatefulPartitionedCall)serving_default_separable_conv2d_12_input$separable_conv2d_12/depthwise_kernel$separable_conv2d_12/pointwise_kernelseparable_conv2d_12/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variance$separable_conv2d_13/depthwise_kernel$separable_conv2d_13/pointwise_kernelseparable_conv2d_13/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variance$separable_conv2d_14/depthwise_kernel$separable_conv2d_14/pointwise_kernelseparable_conv2d_14/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance$separable_conv2d_15/depthwise_kernel$separable_conv2d_15/pointwise_kernelseparable_conv2d_15/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance$separable_conv2d_16/depthwise_kernel$separable_conv2d_16/pointwise_kernelseparable_conv2d_16/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_variance$separable_conv2d_17/depthwise_kernel$separable_conv2d_17/pointwise_kernelseparable_conv2d_17/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_4/kerneldense_4/bias&batch_normalization_20/moving_variancebatch_normalization_20/gamma"batch_normalization_20/moving_meanbatch_normalization_20/betadense_5/kerneldense_5/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_322211
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ń,
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8separable_conv2d_12/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_12/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_12/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp8separable_conv2d_13/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_13/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_13/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp8separable_conv2d_14/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_14/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_14/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp8separable_conv2d_15/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_15/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_15/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp8separable_conv2d_16/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_16/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_16/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp8separable_conv2d_17/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_17/pointwise_kernel/Read/ReadVariableOp,separable_conv2d_17/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpLAdagrad/separable_conv2d_12/depthwise_kernel/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_12/pointwise_kernel/accumulator/Read/ReadVariableOp@Adagrad/separable_conv2d_12/bias/accumulator/Read/ReadVariableOpDAdagrad/batch_normalization_14/gamma/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_14/beta/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_13/depthwise_kernel/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_13/pointwise_kernel/accumulator/Read/ReadVariableOp@Adagrad/separable_conv2d_13/bias/accumulator/Read/ReadVariableOpDAdagrad/batch_normalization_15/gamma/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_15/beta/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_14/depthwise_kernel/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_14/pointwise_kernel/accumulator/Read/ReadVariableOp@Adagrad/separable_conv2d_14/bias/accumulator/Read/ReadVariableOpDAdagrad/batch_normalization_16/gamma/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_16/beta/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_15/depthwise_kernel/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_15/pointwise_kernel/accumulator/Read/ReadVariableOp@Adagrad/separable_conv2d_15/bias/accumulator/Read/ReadVariableOpDAdagrad/batch_normalization_17/gamma/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_17/beta/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_16/depthwise_kernel/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_16/pointwise_kernel/accumulator/Read/ReadVariableOp@Adagrad/separable_conv2d_16/bias/accumulator/Read/ReadVariableOpDAdagrad/batch_normalization_18/gamma/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_18/beta/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_17/depthwise_kernel/accumulator/Read/ReadVariableOpLAdagrad/separable_conv2d_17/pointwise_kernel/accumulator/Read/ReadVariableOp@Adagrad/separable_conv2d_17/bias/accumulator/Read/ReadVariableOpDAdagrad/batch_normalization_19/gamma/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_19/beta/accumulator/Read/ReadVariableOp6Adagrad/dense_4/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_4/bias/accumulator/Read/ReadVariableOpDAdagrad/batch_normalization_20/gamma/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_20/beta/accumulator/Read/ReadVariableOp6Adagrad/dense_5/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_5/bias/accumulator/Read/ReadVariableOpConst*j
Tinc
a2_	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_324215
Ø
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$separable_conv2d_12/depthwise_kernel$separable_conv2d_12/pointwise_kernelseparable_conv2d_12/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variance$separable_conv2d_13/depthwise_kernel$separable_conv2d_13/pointwise_kernelseparable_conv2d_13/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_variance$separable_conv2d_14/depthwise_kernel$separable_conv2d_14/pointwise_kernelseparable_conv2d_14/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_variance$separable_conv2d_15/depthwise_kernel$separable_conv2d_15/pointwise_kernelseparable_conv2d_15/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_variance$separable_conv2d_16/depthwise_kernel$separable_conv2d_16/pointwise_kernelseparable_conv2d_16/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_variance$separable_conv2d_17/depthwise_kernel$separable_conv2d_17/pointwise_kernelseparable_conv2d_17/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_4/kerneldense_4/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_variancedense_5/kerneldense_5/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotalcounttotal_1count_18Adagrad/separable_conv2d_12/depthwise_kernel/accumulator8Adagrad/separable_conv2d_12/pointwise_kernel/accumulator,Adagrad/separable_conv2d_12/bias/accumulator0Adagrad/batch_normalization_14/gamma/accumulator/Adagrad/batch_normalization_14/beta/accumulator8Adagrad/separable_conv2d_13/depthwise_kernel/accumulator8Adagrad/separable_conv2d_13/pointwise_kernel/accumulator,Adagrad/separable_conv2d_13/bias/accumulator0Adagrad/batch_normalization_15/gamma/accumulator/Adagrad/batch_normalization_15/beta/accumulator8Adagrad/separable_conv2d_14/depthwise_kernel/accumulator8Adagrad/separable_conv2d_14/pointwise_kernel/accumulator,Adagrad/separable_conv2d_14/bias/accumulator0Adagrad/batch_normalization_16/gamma/accumulator/Adagrad/batch_normalization_16/beta/accumulator8Adagrad/separable_conv2d_15/depthwise_kernel/accumulator8Adagrad/separable_conv2d_15/pointwise_kernel/accumulator,Adagrad/separable_conv2d_15/bias/accumulator0Adagrad/batch_normalization_17/gamma/accumulator/Adagrad/batch_normalization_17/beta/accumulator8Adagrad/separable_conv2d_16/depthwise_kernel/accumulator8Adagrad/separable_conv2d_16/pointwise_kernel/accumulator,Adagrad/separable_conv2d_16/bias/accumulator0Adagrad/batch_normalization_18/gamma/accumulator/Adagrad/batch_normalization_18/beta/accumulator8Adagrad/separable_conv2d_17/depthwise_kernel/accumulator8Adagrad/separable_conv2d_17/pointwise_kernel/accumulator,Adagrad/separable_conv2d_17/bias/accumulator0Adagrad/batch_normalization_19/gamma/accumulator/Adagrad/batch_normalization_19/beta/accumulator"Adagrad/dense_4/kernel/accumulator Adagrad/dense_4/bias/accumulator0Adagrad/batch_normalization_20/gamma/accumulator/Adagrad/batch_normalization_20/beta/accumulator"Adagrad/dense_5/kernel/accumulator Adagrad/dense_5/bias/accumulator*i
Tinb
`2^*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_324504Ćæ!
Ś
L
0__inference_max_pooling2d_7_layer_call_fn_319880

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3198742
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs


R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_319764

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ģ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

d
F__inference_dropout_10_layer_call_and_return_conditional_losses_320791

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ķ
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_323022

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yĘ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’ :W S
/
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

e
I__inference_activation_14_layer_call_and_return_conditional_losses_320535

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’00 :W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
į

O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_319896

inputsB
(separable_conv2d_readvariableop_resource:@E
*separable_conv2d_readvariableop_1_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOpŗ
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*
paddingSAME*
strides
2
separable_conv2d/depthwiseō
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp„
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2	
BiasAddŽ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

Å
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_321122

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1’
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Ö
7__inference_batch_normalization_20_layer_call_fn_323812

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_3204382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦
Ņ
7__inference_batch_normalization_15_layer_call_fn_323071

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3206032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ķ
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_321332

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yĘ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’ :W S
/
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ų
Ī
4__inference_separable_conv2d_14_layer_call_fn_319742

inputs!
unknown:@#
	unknown_0:@@
	unknown_1:@
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_3197302
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
¬
Ö
7__inference_batch_normalization_19_layer_call_fn_323647

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3210722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Å
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_321072

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1’
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

e
I__inference_activation_16_layer_call_and_return_conditional_losses_323166

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ō
Ö
7__inference_batch_normalization_18_layer_call_fn_323487

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_3201282
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ō
”
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323567

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ū
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
«
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_319874

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ą
Į
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323254

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

”
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_320238

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ķ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ē
G
+__inference_dropout_11_layer_call_fn_323871

inputs
identityČ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3208382
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±¢
É
H__inference_sequential_2_layer_call_and_return_conditional_losses_322102
separable_conv2d_12_input4
separable_conv2d_12_321971:4
separable_conv2d_12_321973: (
separable_conv2d_12_321975: +
batch_normalization_14_321979: +
batch_normalization_14_321981: +
batch_normalization_14_321983: +
batch_normalization_14_321985: 4
separable_conv2d_13_321990: 4
separable_conv2d_13_321992: @(
separable_conv2d_13_321994:@+
batch_normalization_15_321998:@+
batch_normalization_15_322000:@+
batch_normalization_15_322002:@+
batch_normalization_15_322004:@4
separable_conv2d_14_322007:@4
separable_conv2d_14_322009:@@(
separable_conv2d_14_322011:@+
batch_normalization_16_322015:@+
batch_normalization_16_322017:@+
batch_normalization_16_322019:@+
batch_normalization_16_322021:@4
separable_conv2d_15_322026:@5
separable_conv2d_15_322028:@)
separable_conv2d_15_322030:	,
batch_normalization_17_322034:	,
batch_normalization_17_322036:	,
batch_normalization_17_322038:	,
batch_normalization_17_322040:	5
separable_conv2d_16_322043:6
separable_conv2d_16_322045:)
separable_conv2d_16_322047:	,
batch_normalization_18_322051:	,
batch_normalization_18_322053:	,
batch_normalization_18_322055:	,
batch_normalization_18_322057:	5
separable_conv2d_17_322060:6
separable_conv2d_17_322062:)
separable_conv2d_17_322064:	,
batch_normalization_19_322068:	,
batch_normalization_19_322070:	,
batch_normalization_19_322072:	,
batch_normalization_19_322074:	"
dense_4_322080:
$
dense_4_322082:	,
batch_normalization_20_322086:	,
batch_normalization_20_322088:	,
batch_normalization_20_322090:	,
batch_normalization_20_322092:	!
dense_5_322096:	
dense_5_322098:
identity¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢"dropout_10/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢+separable_conv2d_12/StatefulPartitionedCall¢+separable_conv2d_13/StatefulPartitionedCall¢+separable_conv2d_14/StatefulPartitionedCall¢+separable_conv2d_15/StatefulPartitionedCall¢+separable_conv2d_16/StatefulPartitionedCall¢+separable_conv2d_17/StatefulPartitionedCall
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCallseparable_conv2d_12_inputseparable_conv2d_12_321971separable_conv2d_12_321973separable_conv2d_12_321975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_3194102-
+separable_conv2d_12/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_3205352
activation_14/PartitionedCallÅ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0batch_normalization_14_321979batch_normalization_14_321981batch_normalization_14_321983batch_normalization_14_321985*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_32136820
.batch_normalization_14/StatefulPartitionedCall„
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3195542!
max_pooling2d_6/PartitionedCall
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3213322#
!dropout_8/StatefulPartitionedCall
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0separable_conv2d_13_321990separable_conv2d_13_321992separable_conv2d_13_321994*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_3195762-
+separable_conv2d_13/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_3205842
activation_15/PartitionedCallÅ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0batch_normalization_15_321998batch_normalization_15_322000batch_normalization_15_322002batch_normalization_15_322004*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_32129520
.batch_normalization_15/StatefulPartitionedCall„
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0separable_conv2d_14_322007separable_conv2d_14_322009separable_conv2d_14_322011*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_3197302-
+separable_conv2d_14/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_3206252
activation_16/PartitionedCallÅ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_16_322015batch_normalization_16_322017batch_normalization_16_322019batch_normalization_16_322021*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_32124520
.batch_normalization_16/StatefulPartitionedCall„
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3198742!
max_pooling2d_7/PartitionedCallĄ
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3212092#
!dropout_9/StatefulPartitionedCall
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0separable_conv2d_15_322026separable_conv2d_15_322028separable_conv2d_15_322030*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_3198962-
+separable_conv2d_15/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_3206742
activation_17/PartitionedCallĘ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_17_322034batch_normalization_17_322036batch_normalization_17_322038batch_normalization_17_322040*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_32117220
.batch_normalization_17/StatefulPartitionedCall¦
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0separable_conv2d_16_322043separable_conv2d_16_322045separable_conv2d_16_322047*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_3200502-
+separable_conv2d_16/StatefulPartitionedCall
activation_18/PartitionedCallPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_3207152
activation_18/PartitionedCallĘ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_18_322051batch_normalization_18_322053batch_normalization_18_322055batch_normalization_18_322057*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_32112220
.batch_normalization_18/StatefulPartitionedCall¦
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0separable_conv2d_17_322060separable_conv2d_17_322062separable_conv2d_17_322064*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_3202042-
+separable_conv2d_17/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_3207562
activation_19/PartitionedCallĘ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0batch_normalization_19_322068batch_normalization_19_322070batch_normalization_19_322072batch_normalization_19_322074*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_32107220
.batch_normalization_19/StatefulPartitionedCall¦
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3203482!
max_pooling2d_8/PartitionedCallÄ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3210362$
"dropout_10/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3207992
flatten_2/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_322080dense_4_322082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3208112!
dense_4/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_3208222
activation_20/PartitionedCall¾
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0batch_normalization_20_322086batch_normalization_20_322088batch_normalization_20_322090batch_normalization_20_322092*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_32043820
.batch_normalization_20/StatefulPartitionedCallĢ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3209912$
"dropout_11/StatefulPartitionedCall·
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_322096dense_5_322098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3208512!
dense_5/StatefulPartitionedCall½
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall:j f
/
_output_shapes
:’’’’’’’’’00
3
_user_specified_nameseparable_conv2d_12_input
÷
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_323881

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322941

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3ģ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs
¤
Ņ
7__inference_batch_normalization_14_layer_call_fn_322923

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3213682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’00 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
Ō
”
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_320775

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ū
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
·»
1
H__inference_sequential_2_layer_call_and_return_conditional_losses_322620

inputsV
<separable_conv2d_12_separable_conv2d_readvariableop_resource:X
>separable_conv2d_12_separable_conv2d_readvariableop_1_resource: A
3separable_conv2d_12_biasadd_readvariableop_resource: <
.batch_normalization_14_readvariableop_resource: >
0batch_normalization_14_readvariableop_1_resource: M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource: V
<separable_conv2d_13_separable_conv2d_readvariableop_resource: X
>separable_conv2d_13_separable_conv2d_readvariableop_1_resource: @A
3separable_conv2d_13_biasadd_readvariableop_resource:@<
.batch_normalization_15_readvariableop_resource:@>
0batch_normalization_15_readvariableop_1_resource:@M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@V
<separable_conv2d_14_separable_conv2d_readvariableop_resource:@X
>separable_conv2d_14_separable_conv2d_readvariableop_1_resource:@@A
3separable_conv2d_14_biasadd_readvariableop_resource:@<
.batch_normalization_16_readvariableop_resource:@>
0batch_normalization_16_readvariableop_1_resource:@M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@V
<separable_conv2d_15_separable_conv2d_readvariableop_resource:@Y
>separable_conv2d_15_separable_conv2d_readvariableop_1_resource:@B
3separable_conv2d_15_biasadd_readvariableop_resource:	=
.batch_normalization_17_readvariableop_resource:	?
0batch_normalization_17_readvariableop_1_resource:	N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	W
<separable_conv2d_16_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_16_separable_conv2d_readvariableop_1_resource:B
3separable_conv2d_16_biasadd_readvariableop_resource:	=
.batch_normalization_18_readvariableop_resource:	?
0batch_normalization_18_readvariableop_1_resource:	N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	W
<separable_conv2d_17_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_17_separable_conv2d_readvariableop_1_resource:B
3separable_conv2d_17_biasadd_readvariableop_resource:	=
.batch_normalization_19_readvariableop_resource:	?
0batch_normalization_19_readvariableop_1_resource:	N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	:
&dense_4_matmul_readvariableop_resource:
$6
'dense_4_biasadd_readvariableop_resource:	G
8batch_normalization_20_batchnorm_readvariableop_resource:	K
<batch_normalization_20_batchnorm_mul_readvariableop_resource:	I
:batch_normalization_20_batchnorm_readvariableop_1_resource:	I
:batch_normalization_20_batchnorm_readvariableop_2_resource:	9
&dense_5_matmul_readvariableop_resource:	5
'dense_5_biasadd_readvariableop_resource:
identity¢6batch_normalization_14/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_14/ReadVariableOp¢'batch_normalization_14/ReadVariableOp_1¢6batch_normalization_15/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_15/ReadVariableOp¢'batch_normalization_15/ReadVariableOp_1¢6batch_normalization_16/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_16/ReadVariableOp¢'batch_normalization_16/ReadVariableOp_1¢6batch_normalization_17/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_17/ReadVariableOp¢'batch_normalization_17/ReadVariableOp_1¢6batch_normalization_18/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_18/ReadVariableOp¢'batch_normalization_18/ReadVariableOp_1¢6batch_normalization_19/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_19/ReadVariableOp¢'batch_normalization_19/ReadVariableOp_1¢/batch_normalization_20/batchnorm/ReadVariableOp¢1batch_normalization_20/batchnorm/ReadVariableOp_1¢1batch_normalization_20/batchnorm/ReadVariableOp_2¢3batch_normalization_20/batchnorm/mul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢*separable_conv2d_12/BiasAdd/ReadVariableOp¢3separable_conv2d_12/separable_conv2d/ReadVariableOp¢5separable_conv2d_12/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_13/BiasAdd/ReadVariableOp¢3separable_conv2d_13/separable_conv2d/ReadVariableOp¢5separable_conv2d_13/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_14/BiasAdd/ReadVariableOp¢3separable_conv2d_14/separable_conv2d/ReadVariableOp¢5separable_conv2d_14/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_15/BiasAdd/ReadVariableOp¢3separable_conv2d_15/separable_conv2d/ReadVariableOp¢5separable_conv2d_15/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_16/BiasAdd/ReadVariableOp¢3separable_conv2d_16/separable_conv2d/ReadVariableOp¢5separable_conv2d_16/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_17/BiasAdd/ReadVariableOp¢3separable_conv2d_17/separable_conv2d/ReadVariableOp¢5separable_conv2d_17/separable_conv2d/ReadVariableOp_1ļ
3separable_conv2d_12/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_12_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3separable_conv2d_12/separable_conv2d/ReadVariableOpõ
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_12_separable_conv2d_readvariableop_1_resource*&
_output_shapes
: *
dtype027
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_12/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_12/separable_conv2d/Shape¹
2separable_conv2d_12/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_12/separable_conv2d/dilation_rate 
.separable_conv2d_12/separable_conv2d/depthwiseDepthwiseConv2dNativeinputs;separable_conv2d_12/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’00*
paddingSAME*
strides
20
.separable_conv2d_12/separable_conv2d/depthwise±
$separable_conv2d_12/separable_conv2dConv2D7separable_conv2d_12/separable_conv2d/depthwise:output:0=separable_conv2d_12/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’00 *
paddingVALID*
strides
2&
$separable_conv2d_12/separable_conv2dČ
*separable_conv2d_12/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*separable_conv2d_12/BiasAdd/ReadVariableOpā
separable_conv2d_12/BiasAddBiasAdd-separable_conv2d_12/separable_conv2d:output:02separable_conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’00 2
separable_conv2d_12/BiasAdd
activation_14/ReluRelu$separable_conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’00 2
activation_14/Relu¹
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_14/ReadVariableOpæ
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_14/ReadVariableOp_1ģ
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpņ
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ī
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3 activation_14/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’00 : : : : :*
epsilon%o:*
is_training( 2)
'batch_normalization_14/FusedBatchNormV3×
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_14/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool
dropout_8/IdentityIdentity max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout_8/Identityļ
3separable_conv2d_13/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_13_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype025
3separable_conv2d_13/separable_conv2d/ReadVariableOpõ
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_13_separable_conv2d_readvariableop_1_resource*&
_output_shapes
: @*
dtype027
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_13/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*separable_conv2d_13/separable_conv2d/Shape¹
2separable_conv2d_13/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_13/separable_conv2d/dilation_rateµ
.separable_conv2d_13/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_8/Identity:output:0;separable_conv2d_13/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’ *
paddingSAME*
strides
20
.separable_conv2d_13/separable_conv2d/depthwise±
$separable_conv2d_13/separable_conv2dConv2D7separable_conv2d_13/separable_conv2d/depthwise:output:0=separable_conv2d_13/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
2&
$separable_conv2d_13/separable_conv2dČ
*separable_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*separable_conv2d_13/BiasAdd/ReadVariableOpā
separable_conv2d_13/BiasAddBiasAdd-separable_conv2d_13/separable_conv2d:output:02separable_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@2
separable_conv2d_13/BiasAdd
activation_15/ReluRelu$separable_conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
activation_15/Relu¹
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOpæ
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1ģ
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpņ
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ī
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3 activation_15/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3ļ
3separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_14_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype025
3separable_conv2d_14/separable_conv2d/ReadVariableOpõ
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_14_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@@*
dtype027
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2,
*separable_conv2d_14/separable_conv2d/Shape¹
2separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_14/separable_conv2d/dilation_rateÅ
.separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNative+batch_normalization_15/FusedBatchNormV3:y:0;separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
20
.separable_conv2d_14/separable_conv2d/depthwise±
$separable_conv2d_14/separable_conv2dConv2D7separable_conv2d_14/separable_conv2d/depthwise:output:0=separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
2&
$separable_conv2d_14/separable_conv2dČ
*separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*separable_conv2d_14/BiasAdd/ReadVariableOpā
separable_conv2d_14/BiasAddBiasAdd-separable_conv2d_14/separable_conv2d:output:02separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@2
separable_conv2d_14/BiasAdd
activation_16/ReluRelu$separable_conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
activation_16/Relu¹
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOpæ
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1ģ
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpņ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ī
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 activation_16/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3×
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool
dropout_9/IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout_9/Identityļ
3separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_15_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype025
3separable_conv2d_15/separable_conv2d/ReadVariableOpö
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_15_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype027
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2,
*separable_conv2d_15/separable_conv2d/Shape¹
2separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_15/separable_conv2d/dilation_rateµ
.separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_9/Identity:output:0;separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
20
.separable_conv2d_15/separable_conv2d/depthwise²
$separable_conv2d_15/separable_conv2dConv2D7separable_conv2d_15/separable_conv2d/depthwise:output:0=separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2&
$separable_conv2d_15/separable_conv2dÉ
*separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*separable_conv2d_15/BiasAdd/ReadVariableOpć
separable_conv2d_15/BiasAddBiasAdd-separable_conv2d_15/separable_conv2d:output:02separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
separable_conv2d_15/BiasAdd
activation_17/ReluRelu$separable_conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
activation_17/Reluŗ
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_17/ReadVariableOpĄ
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_17/ReadVariableOp_1ķ
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ó
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 activation_17/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3š
3separable_conv2d_16/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_16_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype025
3separable_conv2d_16/separable_conv2d/ReadVariableOp÷
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_16_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype027
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_16/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_16/separable_conv2d/Shape¹
2separable_conv2d_16/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_16/separable_conv2d/dilation_rateĘ
.separable_conv2d_16/separable_conv2d/depthwiseDepthwiseConv2dNative+batch_normalization_17/FusedBatchNormV3:y:0;separable_conv2d_16/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
20
.separable_conv2d_16/separable_conv2d/depthwise²
$separable_conv2d_16/separable_conv2dConv2D7separable_conv2d_16/separable_conv2d/depthwise:output:0=separable_conv2d_16/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2&
$separable_conv2d_16/separable_conv2dÉ
*separable_conv2d_16/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*separable_conv2d_16/BiasAdd/ReadVariableOpć
separable_conv2d_16/BiasAddBiasAdd-separable_conv2d_16/separable_conv2d:output:02separable_conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
separable_conv2d_16/BiasAdd
activation_18/ReluRelu$separable_conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
activation_18/Reluŗ
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_18/ReadVariableOpĄ
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_18/ReadVariableOp_1ķ
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ó
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 activation_18/Relu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_18/FusedBatchNormV3š
3separable_conv2d_17/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_17_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype025
3separable_conv2d_17/separable_conv2d/ReadVariableOp÷
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_17_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype027
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_17/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_17/separable_conv2d/Shape¹
2separable_conv2d_17/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_17/separable_conv2d/dilation_rateĘ
.separable_conv2d_17/separable_conv2d/depthwiseDepthwiseConv2dNative+batch_normalization_18/FusedBatchNormV3:y:0;separable_conv2d_17/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
20
.separable_conv2d_17/separable_conv2d/depthwise²
$separable_conv2d_17/separable_conv2dConv2D7separable_conv2d_17/separable_conv2d/depthwise:output:0=separable_conv2d_17/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2&
$separable_conv2d_17/separable_conv2dÉ
*separable_conv2d_17/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*separable_conv2d_17/BiasAdd/ReadVariableOpć
separable_conv2d_17/BiasAddBiasAdd-separable_conv2d_17/separable_conv2d:output:02separable_conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
separable_conv2d_17/BiasAdd
activation_19/ReluRelu$separable_conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
activation_19/Reluŗ
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_19/ReadVariableOpĄ
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_19/ReadVariableOp_1ķ
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ó
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3 activation_19/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_19/FusedBatchNormV3Ų
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_19/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPool
dropout_10/IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_10/Identitys
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
flatten_2/Const
flatten_2/ReshapeReshapedropout_10/Identity:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2
flatten_2/Reshape§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
$*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/MatMul„
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/BiasAdd}
activation_20/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation_20/ReluŲ
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOp
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_20/batchnorm/add/yå
$batch_normalization_20/batchnorm/addAddV27batch_normalization_20/batchnorm/ReadVariableOp:value:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_20/batchnorm/add©
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_20/batchnorm/Rsqrtä
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOpā
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_20/batchnorm/mulÖ
&batch_normalization_20/batchnorm/mul_1Mul activation_20/Relu:activations:0(batch_normalization_20/batchnorm/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2(
&batch_normalization_20/batchnorm/mul_1Ž
1batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_1ā
&batch_normalization_20/batchnorm/mul_2Mul9batch_normalization_20/batchnorm/ReadVariableOp_1:value:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_20/batchnorm/mul_2Ž
1batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype023
1batch_normalization_20/batchnorm/ReadVariableOp_2ą
$batch_normalization_20/batchnorm/subSub9batch_normalization_20/batchnorm/ReadVariableOp_2:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_20/batchnorm/subā
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2(
&batch_normalization_20/batchnorm/add_1
dropout_11/IdentityIdentity*batch_normalization_20/batchnorm/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_11/Identity¦
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_5/MatMul/ReadVariableOp”
dense_5/MatMulMatMuldropout_11/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp”
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_5/Sigmoid’
IdentityIdentitydense_5/Sigmoid:y:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_10^batch_normalization_20/batchnorm/ReadVariableOp2^batch_normalization_20/batchnorm/ReadVariableOp_12^batch_normalization_20/batchnorm/ReadVariableOp_24^batch_normalization_20/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp+^separable_conv2d_12/BiasAdd/ReadVariableOp4^separable_conv2d_12/separable_conv2d/ReadVariableOp6^separable_conv2d_12/separable_conv2d/ReadVariableOp_1+^separable_conv2d_13/BiasAdd/ReadVariableOp4^separable_conv2d_13/separable_conv2d/ReadVariableOp6^separable_conv2d_13/separable_conv2d/ReadVariableOp_1+^separable_conv2d_14/BiasAdd/ReadVariableOp4^separable_conv2d_14/separable_conv2d/ReadVariableOp6^separable_conv2d_14/separable_conv2d/ReadVariableOp_1+^separable_conv2d_15/BiasAdd/ReadVariableOp4^separable_conv2d_15/separable_conv2d/ReadVariableOp6^separable_conv2d_15/separable_conv2d/ReadVariableOp_1+^separable_conv2d_16/BiasAdd/ReadVariableOp4^separable_conv2d_16/separable_conv2d/ReadVariableOp6^separable_conv2d_16/separable_conv2d/ReadVariableOp_1+^separable_conv2d_17/BiasAdd/ReadVariableOp4^separable_conv2d_17/separable_conv2d/ReadVariableOp6^separable_conv2d_17/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12b
/batch_normalization_20/batchnorm/ReadVariableOp/batch_normalization_20/batchnorm/ReadVariableOp2f
1batch_normalization_20/batchnorm/ReadVariableOp_11batch_normalization_20/batchnorm/ReadVariableOp_12f
1batch_normalization_20/batchnorm/ReadVariableOp_21batch_normalization_20/batchnorm/ReadVariableOp_22j
3batch_normalization_20/batchnorm/mul/ReadVariableOp3batch_normalization_20/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2X
*separable_conv2d_12/BiasAdd/ReadVariableOp*separable_conv2d_12/BiasAdd/ReadVariableOp2j
3separable_conv2d_12/separable_conv2d/ReadVariableOp3separable_conv2d_12/separable_conv2d/ReadVariableOp2n
5separable_conv2d_12/separable_conv2d/ReadVariableOp_15separable_conv2d_12/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_13/BiasAdd/ReadVariableOp*separable_conv2d_13/BiasAdd/ReadVariableOp2j
3separable_conv2d_13/separable_conv2d/ReadVariableOp3separable_conv2d_13/separable_conv2d/ReadVariableOp2n
5separable_conv2d_13/separable_conv2d/ReadVariableOp_15separable_conv2d_13/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_14/BiasAdd/ReadVariableOp*separable_conv2d_14/BiasAdd/ReadVariableOp2j
3separable_conv2d_14/separable_conv2d/ReadVariableOp3separable_conv2d_14/separable_conv2d/ReadVariableOp2n
5separable_conv2d_14/separable_conv2d/ReadVariableOp_15separable_conv2d_14/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_15/BiasAdd/ReadVariableOp*separable_conv2d_15/BiasAdd/ReadVariableOp2j
3separable_conv2d_15/separable_conv2d/ReadVariableOp3separable_conv2d_15/separable_conv2d/ReadVariableOp2n
5separable_conv2d_15/separable_conv2d/ReadVariableOp_15separable_conv2d_15/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_16/BiasAdd/ReadVariableOp*separable_conv2d_16/BiasAdd/ReadVariableOp2j
3separable_conv2d_16/separable_conv2d/ReadVariableOp3separable_conv2d_16/separable_conv2d/ReadVariableOp2n
5separable_conv2d_16/separable_conv2d/ReadVariableOp_15separable_conv2d_16/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_17/BiasAdd/ReadVariableOp*separable_conv2d_17/BiasAdd/ReadVariableOp2j
3separable_conv2d_17/separable_conv2d/ReadVariableOp3separable_conv2d_17/separable_conv2d/ReadVariableOp2n
5separable_conv2d_17/separable_conv2d/ReadVariableOp_15separable_conv2d_17/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
±
µ
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_320378

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
Į
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_321245

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ž
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ü
;
!__inference__wrapped_model_319394
separable_conv2d_12_inputc
Isequential_2_separable_conv2d_12_separable_conv2d_readvariableop_resource:e
Ksequential_2_separable_conv2d_12_separable_conv2d_readvariableop_1_resource: N
@sequential_2_separable_conv2d_12_biasadd_readvariableop_resource: I
;sequential_2_batch_normalization_14_readvariableop_resource: K
=sequential_2_batch_normalization_14_readvariableop_1_resource: Z
Lsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource: \
Nsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource: c
Isequential_2_separable_conv2d_13_separable_conv2d_readvariableop_resource: e
Ksequential_2_separable_conv2d_13_separable_conv2d_readvariableop_1_resource: @N
@sequential_2_separable_conv2d_13_biasadd_readvariableop_resource:@I
;sequential_2_batch_normalization_15_readvariableop_resource:@K
=sequential_2_batch_normalization_15_readvariableop_1_resource:@Z
Lsequential_2_batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_2_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@c
Isequential_2_separable_conv2d_14_separable_conv2d_readvariableop_resource:@e
Ksequential_2_separable_conv2d_14_separable_conv2d_readvariableop_1_resource:@@N
@sequential_2_separable_conv2d_14_biasadd_readvariableop_resource:@I
;sequential_2_batch_normalization_16_readvariableop_resource:@K
=sequential_2_batch_normalization_16_readvariableop_1_resource:@Z
Lsequential_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@c
Isequential_2_separable_conv2d_15_separable_conv2d_readvariableop_resource:@f
Ksequential_2_separable_conv2d_15_separable_conv2d_readvariableop_1_resource:@O
@sequential_2_separable_conv2d_15_biasadd_readvariableop_resource:	J
;sequential_2_batch_normalization_17_readvariableop_resource:	L
=sequential_2_batch_normalization_17_readvariableop_1_resource:	[
Lsequential_2_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_2_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	d
Isequential_2_separable_conv2d_16_separable_conv2d_readvariableop_resource:g
Ksequential_2_separable_conv2d_16_separable_conv2d_readvariableop_1_resource:O
@sequential_2_separable_conv2d_16_biasadd_readvariableop_resource:	J
;sequential_2_batch_normalization_18_readvariableop_resource:	L
=sequential_2_batch_normalization_18_readvariableop_1_resource:	[
Lsequential_2_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_2_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	d
Isequential_2_separable_conv2d_17_separable_conv2d_readvariableop_resource:g
Ksequential_2_separable_conv2d_17_separable_conv2d_readvariableop_1_resource:O
@sequential_2_separable_conv2d_17_biasadd_readvariableop_resource:	J
;sequential_2_batch_normalization_19_readvariableop_resource:	L
=sequential_2_batch_normalization_19_readvariableop_1_resource:	[
Lsequential_2_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_2_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	G
3sequential_2_dense_4_matmul_readvariableop_resource:
$C
4sequential_2_dense_4_biasadd_readvariableop_resource:	T
Esequential_2_batch_normalization_20_batchnorm_readvariableop_resource:	X
Isequential_2_batch_normalization_20_batchnorm_mul_readvariableop_resource:	V
Gsequential_2_batch_normalization_20_batchnorm_readvariableop_1_resource:	V
Gsequential_2_batch_normalization_20_batchnorm_readvariableop_2_resource:	F
3sequential_2_dense_5_matmul_readvariableop_resource:	B
4sequential_2_dense_5_biasadd_readvariableop_resource:
identity¢Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp¢Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1¢2sequential_2/batch_normalization_14/ReadVariableOp¢4sequential_2/batch_normalization_14/ReadVariableOp_1¢Csequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp¢Esequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1¢2sequential_2/batch_normalization_15/ReadVariableOp¢4sequential_2/batch_normalization_15/ReadVariableOp_1¢Csequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp¢Esequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1¢2sequential_2/batch_normalization_16/ReadVariableOp¢4sequential_2/batch_normalization_16/ReadVariableOp_1¢Csequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp¢Esequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1¢2sequential_2/batch_normalization_17/ReadVariableOp¢4sequential_2/batch_normalization_17/ReadVariableOp_1¢Csequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp¢Esequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1¢2sequential_2/batch_normalization_18/ReadVariableOp¢4sequential_2/batch_normalization_18/ReadVariableOp_1¢Csequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp¢Esequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1¢2sequential_2/batch_normalization_19/ReadVariableOp¢4sequential_2/batch_normalization_19/ReadVariableOp_1¢<sequential_2/batch_normalization_20/batchnorm/ReadVariableOp¢>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_1¢>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_2¢@sequential_2/batch_normalization_20/batchnorm/mul/ReadVariableOp¢+sequential_2/dense_4/BiasAdd/ReadVariableOp¢*sequential_2/dense_4/MatMul/ReadVariableOp¢+sequential_2/dense_5/BiasAdd/ReadVariableOp¢*sequential_2/dense_5/MatMul/ReadVariableOp¢7sequential_2/separable_conv2d_12/BiasAdd/ReadVariableOp¢@sequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp¢Bsequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp_1¢7sequential_2/separable_conv2d_13/BiasAdd/ReadVariableOp¢@sequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp¢Bsequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp_1¢7sequential_2/separable_conv2d_14/BiasAdd/ReadVariableOp¢@sequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp¢Bsequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1¢7sequential_2/separable_conv2d_15/BiasAdd/ReadVariableOp¢@sequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp¢Bsequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1¢7sequential_2/separable_conv2d_16/BiasAdd/ReadVariableOp¢@sequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp¢Bsequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1¢7sequential_2/separable_conv2d_17/BiasAdd/ReadVariableOp¢@sequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp¢Bsequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1
@sequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOpReadVariableOpIsequential_2_separable_conv2d_12_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02B
@sequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp
Bsequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp_1ReadVariableOpKsequential_2_separable_conv2d_12_separable_conv2d_readvariableop_1_resource*&
_output_shapes
: *
dtype02D
Bsequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp_1Ė
7sequential_2/separable_conv2d_12/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7sequential_2/separable_conv2d_12/separable_conv2d/ShapeÓ
?sequential_2/separable_conv2d_12/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_2/separable_conv2d_12/separable_conv2d/dilation_rateŚ
;sequential_2/separable_conv2d_12/separable_conv2d/depthwiseDepthwiseConv2dNativeseparable_conv2d_12_inputHsequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’00*
paddingSAME*
strides
2=
;sequential_2/separable_conv2d_12/separable_conv2d/depthwiseå
1sequential_2/separable_conv2d_12/separable_conv2dConv2DDsequential_2/separable_conv2d_12/separable_conv2d/depthwise:output:0Jsequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’00 *
paddingVALID*
strides
23
1sequential_2/separable_conv2d_12/separable_conv2dļ
7sequential_2/separable_conv2d_12/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_separable_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_2/separable_conv2d_12/BiasAdd/ReadVariableOp
(sequential_2/separable_conv2d_12/BiasAddBiasAdd:sequential_2/separable_conv2d_12/separable_conv2d:output:0?sequential_2/separable_conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’00 2*
(sequential_2/separable_conv2d_12/BiasAdd·
sequential_2/activation_14/ReluRelu1sequential_2/separable_conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’00 2!
sequential_2/activation_14/Reluą
2sequential_2/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_2/batch_normalization_14/ReadVariableOpę
4sequential_2/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_2/batch_normalization_14/ReadVariableOp_1
Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02E
Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp
Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1É
4sequential_2/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3-sequential_2/activation_14/Relu:activations:0:sequential_2/batch_normalization_14/ReadVariableOp:value:0<sequential_2/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’00 : : : : :*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_14/FusedBatchNormV3ž
$sequential_2/max_pooling2d_6/MaxPoolMaxPool8sequential_2/batch_normalization_14/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’ *
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_6/MaxPool·
sequential_2/dropout_8/IdentityIdentity-sequential_2/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2!
sequential_2/dropout_8/Identity
@sequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOpReadVariableOpIsequential_2_separable_conv2d_13_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02B
@sequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp
Bsequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp_1ReadVariableOpKsequential_2_separable_conv2d_13_separable_conv2d_readvariableop_1_resource*&
_output_shapes
: @*
dtype02D
Bsequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp_1Ė
7sequential_2/separable_conv2d_13/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             29
7sequential_2/separable_conv2d_13/separable_conv2d/ShapeÓ
?sequential_2/separable_conv2d_13/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_2/separable_conv2d_13/separable_conv2d/dilation_rateé
;sequential_2/separable_conv2d_13/separable_conv2d/depthwiseDepthwiseConv2dNative(sequential_2/dropout_8/Identity:output:0Hsequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’ *
paddingSAME*
strides
2=
;sequential_2/separable_conv2d_13/separable_conv2d/depthwiseå
1sequential_2/separable_conv2d_13/separable_conv2dConv2DDsequential_2/separable_conv2d_13/separable_conv2d/depthwise:output:0Jsequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
23
1sequential_2/separable_conv2d_13/separable_conv2dļ
7sequential_2/separable_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_separable_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_2/separable_conv2d_13/BiasAdd/ReadVariableOp
(sequential_2/separable_conv2d_13/BiasAddBiasAdd:sequential_2/separable_conv2d_13/separable_conv2d:output:0?sequential_2/separable_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@2*
(sequential_2/separable_conv2d_13/BiasAdd·
sequential_2/activation_15/ReluRelu1sequential_2/separable_conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2!
sequential_2/activation_15/Reluą
2sequential_2/batch_normalization_15/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_2/batch_normalization_15/ReadVariableOpę
4sequential_2/batch_normalization_15/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_2/batch_normalization_15/ReadVariableOp_1
Csequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp
Esequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1É
4sequential_2/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3-sequential_2/activation_15/Relu:activations:0:sequential_2/batch_normalization_15/ReadVariableOp:value:0<sequential_2/batch_normalization_15/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_15/FusedBatchNormV3
@sequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOpIsequential_2_separable_conv2d_14_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02B
@sequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp
Bsequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOpKsequential_2_separable_conv2d_14_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@@*
dtype02D
Bsequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1Ė
7sequential_2/separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      29
7sequential_2/separable_conv2d_14/separable_conv2d/ShapeÓ
?sequential_2/separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_2/separable_conv2d_14/separable_conv2d/dilation_rateł
;sequential_2/separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNative8sequential_2/batch_normalization_15/FusedBatchNormV3:y:0Hsequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2=
;sequential_2/separable_conv2d_14/separable_conv2d/depthwiseå
1sequential_2/separable_conv2d_14/separable_conv2dConv2DDsequential_2/separable_conv2d_14/separable_conv2d/depthwise:output:0Jsequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
23
1sequential_2/separable_conv2d_14/separable_conv2dļ
7sequential_2/separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_2/separable_conv2d_14/BiasAdd/ReadVariableOp
(sequential_2/separable_conv2d_14/BiasAddBiasAdd:sequential_2/separable_conv2d_14/separable_conv2d:output:0?sequential_2/separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@2*
(sequential_2/separable_conv2d_14/BiasAdd·
sequential_2/activation_16/ReluRelu1sequential_2/separable_conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2!
sequential_2/activation_16/Reluą
2sequential_2/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype024
2sequential_2/batch_normalization_16/ReadVariableOpę
4sequential_2/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype026
4sequential_2/batch_normalization_16/ReadVariableOp_1
Csequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp
Esequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
Esequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1É
4sequential_2/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3-sequential_2/activation_16/Relu:activations:0:sequential_2/batch_normalization_16/ReadVariableOp:value:0<sequential_2/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_16/FusedBatchNormV3ž
$sequential_2/max_pooling2d_7/MaxPoolMaxPool8sequential_2/batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’@*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_7/MaxPool·
sequential_2/dropout_9/IdentityIdentity-sequential_2/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2!
sequential_2/dropout_9/Identity
@sequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOpIsequential_2_separable_conv2d_15_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02B
@sequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp
Bsequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOpKsequential_2_separable_conv2d_15_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype02D
Bsequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1Ė
7sequential_2/separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      29
7sequential_2/separable_conv2d_15/separable_conv2d/ShapeÓ
?sequential_2/separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_2/separable_conv2d_15/separable_conv2d/dilation_rateé
;sequential_2/separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNative(sequential_2/dropout_9/Identity:output:0Hsequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
2=
;sequential_2/separable_conv2d_15/separable_conv2d/depthwiseę
1sequential_2/separable_conv2d_15/separable_conv2dConv2DDsequential_2/separable_conv2d_15/separable_conv2d/depthwise:output:0Jsequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
23
1sequential_2/separable_conv2d_15/separable_conv2dš
7sequential_2/separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential_2/separable_conv2d_15/BiasAdd/ReadVariableOp
(sequential_2/separable_conv2d_15/BiasAddBiasAdd:sequential_2/separable_conv2d_15/separable_conv2d:output:0?sequential_2/separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2*
(sequential_2/separable_conv2d_15/BiasAddø
sequential_2/activation_17/ReluRelu1sequential_2/separable_conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2!
sequential_2/activation_17/Reluį
2sequential_2/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_17_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_2/batch_normalization_17/ReadVariableOpē
4sequential_2/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4sequential_2/batch_normalization_17/ReadVariableOp_1
Csequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Csequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp
Esequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Esequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Ī
4sequential_2/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3-sequential_2/activation_17/Relu:activations:0:sequential_2/batch_normalization_17/ReadVariableOp:value:0<sequential_2/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_17/FusedBatchNormV3
@sequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOpReadVariableOpIsequential_2_separable_conv2d_16_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02B
@sequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp
Bsequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1ReadVariableOpKsequential_2_separable_conv2d_16_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype02D
Bsequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1Ė
7sequential_2/separable_conv2d_16/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7sequential_2/separable_conv2d_16/separable_conv2d/ShapeÓ
?sequential_2/separable_conv2d_16/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_2/separable_conv2d_16/separable_conv2d/dilation_rateś
;sequential_2/separable_conv2d_16/separable_conv2d/depthwiseDepthwiseConv2dNative8sequential_2/batch_normalization_17/FusedBatchNormV3:y:0Hsequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2=
;sequential_2/separable_conv2d_16/separable_conv2d/depthwiseę
1sequential_2/separable_conv2d_16/separable_conv2dConv2DDsequential_2/separable_conv2d_16/separable_conv2d/depthwise:output:0Jsequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
23
1sequential_2/separable_conv2d_16/separable_conv2dš
7sequential_2/separable_conv2d_16/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_separable_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential_2/separable_conv2d_16/BiasAdd/ReadVariableOp
(sequential_2/separable_conv2d_16/BiasAddBiasAdd:sequential_2/separable_conv2d_16/separable_conv2d:output:0?sequential_2/separable_conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2*
(sequential_2/separable_conv2d_16/BiasAddø
sequential_2/activation_18/ReluRelu1sequential_2/separable_conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2!
sequential_2/activation_18/Reluį
2sequential_2/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_18_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_2/batch_normalization_18/ReadVariableOpē
4sequential_2/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4sequential_2/batch_normalization_18/ReadVariableOp_1
Csequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Csequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp
Esequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Esequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Ī
4sequential_2/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3-sequential_2/activation_18/Relu:activations:0:sequential_2/batch_normalization_18/ReadVariableOp:value:0<sequential_2/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_18/FusedBatchNormV3
@sequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOpReadVariableOpIsequential_2_separable_conv2d_17_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02B
@sequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp
Bsequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1ReadVariableOpKsequential_2_separable_conv2d_17_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype02D
Bsequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1Ė
7sequential_2/separable_conv2d_17/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            29
7sequential_2/separable_conv2d_17/separable_conv2d/ShapeÓ
?sequential_2/separable_conv2d_17/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential_2/separable_conv2d_17/separable_conv2d/dilation_rateś
;sequential_2/separable_conv2d_17/separable_conv2d/depthwiseDepthwiseConv2dNative8sequential_2/batch_normalization_18/FusedBatchNormV3:y:0Hsequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
2=
;sequential_2/separable_conv2d_17/separable_conv2d/depthwiseę
1sequential_2/separable_conv2d_17/separable_conv2dConv2DDsequential_2/separable_conv2d_17/separable_conv2d/depthwise:output:0Jsequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
23
1sequential_2/separable_conv2d_17/separable_conv2dš
7sequential_2/separable_conv2d_17/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_separable_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential_2/separable_conv2d_17/BiasAdd/ReadVariableOp
(sequential_2/separable_conv2d_17/BiasAddBiasAdd:sequential_2/separable_conv2d_17/separable_conv2d:output:0?sequential_2/separable_conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2*
(sequential_2/separable_conv2d_17/BiasAddø
sequential_2/activation_19/ReluRelu1sequential_2/separable_conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2!
sequential_2/activation_19/Reluį
2sequential_2/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_19_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_2/batch_normalization_19/ReadVariableOpē
4sequential_2/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:*
dtype026
4sequential_2/batch_normalization_19/ReadVariableOp_1
Csequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02E
Csequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp
Esequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
Esequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Ī
4sequential_2/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3-sequential_2/activation_19/Relu:activations:0:sequential_2/batch_normalization_19/ReadVariableOp:value:0<sequential_2/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 26
4sequential_2/batch_normalization_19/FusedBatchNormV3’
$sequential_2/max_pooling2d_8/MaxPoolMaxPool8sequential_2/batch_normalization_19/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2&
$sequential_2/max_pooling2d_8/MaxPoolŗ
 sequential_2/dropout_10/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:’’’’’’’’’2"
 sequential_2/dropout_10/Identity
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
sequential_2/flatten_2/ConstŠ
sequential_2/flatten_2/ReshapeReshape)sequential_2/dropout_10/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2 
sequential_2/flatten_2/ReshapeĪ
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
$*
dtype02,
*sequential_2/dense_4/MatMul/ReadVariableOpŌ
sequential_2/dense_4/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_2/dense_4/MatMulĢ
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_2/dense_4/BiasAdd/ReadVariableOpÖ
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential_2/dense_4/BiasAdd¤
sequential_2/activation_20/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2!
sequential_2/activation_20/Relu’
<sequential_2/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOpEsequential_2_batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02>
<sequential_2/batch_normalization_20/batchnorm/ReadVariableOpÆ
3sequential_2/batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3sequential_2/batch_normalization_20/batchnorm/add/y
1sequential_2/batch_normalization_20/batchnorm/addAddV2Dsequential_2/batch_normalization_20/batchnorm/ReadVariableOp:value:0<sequential_2/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:23
1sequential_2/batch_normalization_20/batchnorm/addŠ
3sequential_2/batch_normalization_20/batchnorm/RsqrtRsqrt5sequential_2/batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:25
3sequential_2/batch_normalization_20/batchnorm/Rsqrt
@sequential_2/batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_2_batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02B
@sequential_2/batch_normalization_20/batchnorm/mul/ReadVariableOp
1sequential_2/batch_normalization_20/batchnorm/mulMul7sequential_2/batch_normalization_20/batchnorm/Rsqrt:y:0Hsequential_2/batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:23
1sequential_2/batch_normalization_20/batchnorm/mul
3sequential_2/batch_normalization_20/batchnorm/mul_1Mul-sequential_2/activation_20/Relu:activations:05sequential_2/batch_normalization_20/batchnorm/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’25
3sequential_2/batch_normalization_20/batchnorm/mul_1
>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_2_batch_normalization_20_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02@
>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_1
3sequential_2/batch_normalization_20/batchnorm/mul_2MulFsequential_2/batch_normalization_20/batchnorm/ReadVariableOp_1:value:05sequential_2/batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:25
3sequential_2/batch_normalization_20/batchnorm/mul_2
>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_2_batch_normalization_20_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02@
>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_2
1sequential_2/batch_normalization_20/batchnorm/subSubFsequential_2/batch_normalization_20/batchnorm/ReadVariableOp_2:value:07sequential_2/batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:23
1sequential_2/batch_normalization_20/batchnorm/sub
3sequential_2/batch_normalization_20/batchnorm/add_1AddV27sequential_2/batch_normalization_20/batchnorm/mul_1:z:05sequential_2/batch_normalization_20/batchnorm/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’25
3sequential_2/batch_normalization_20/batchnorm/add_1¼
 sequential_2/dropout_11/IdentityIdentity7sequential_2/batch_normalization_20/batchnorm/add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2"
 sequential_2/dropout_11/IdentityĶ
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_2/dense_5/MatMul/ReadVariableOpÕ
sequential_2/dense_5/MatMulMatMul)sequential_2/dropout_11/Identity:output:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_5/MatMulĖ
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_5/BiasAdd/ReadVariableOpÕ
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_5/BiasAdd 
sequential_2/dense_5/SigmoidSigmoid%sequential_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential_2/dense_5/Sigmoid
IdentityIdentity sequential_2/dense_5/Sigmoid:y:0D^sequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_14/ReadVariableOp5^sequential_2/batch_normalization_14/ReadVariableOp_1D^sequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_15/ReadVariableOp5^sequential_2/batch_normalization_15/ReadVariableOp_1D^sequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_16/ReadVariableOp5^sequential_2/batch_normalization_16/ReadVariableOp_1D^sequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_17/ReadVariableOp5^sequential_2/batch_normalization_17/ReadVariableOp_1D^sequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_18/ReadVariableOp5^sequential_2/batch_normalization_18/ReadVariableOp_1D^sequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_19/ReadVariableOp5^sequential_2/batch_normalization_19/ReadVariableOp_1=^sequential_2/batch_normalization_20/batchnorm/ReadVariableOp?^sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_1?^sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_2A^sequential_2/batch_normalization_20/batchnorm/mul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOp8^sequential_2/separable_conv2d_12/BiasAdd/ReadVariableOpA^sequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOpC^sequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp_18^sequential_2/separable_conv2d_13/BiasAdd/ReadVariableOpA^sequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOpC^sequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp_18^sequential_2/separable_conv2d_14/BiasAdd/ReadVariableOpA^sequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOpC^sequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_18^sequential_2/separable_conv2d_15/BiasAdd/ReadVariableOpA^sequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOpC^sequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_18^sequential_2/separable_conv2d_16/BiasAdd/ReadVariableOpA^sequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOpC^sequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_18^sequential_2/separable_conv2d_17/BiasAdd/ReadVariableOpA^sequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOpC^sequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_14/ReadVariableOp2sequential_2/batch_normalization_14/ReadVariableOp2l
4sequential_2/batch_normalization_14/ReadVariableOp_14sequential_2/batch_normalization_14/ReadVariableOp_12
Csequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_15/ReadVariableOp2sequential_2/batch_normalization_15/ReadVariableOp2l
4sequential_2/batch_normalization_15/ReadVariableOp_14sequential_2/batch_normalization_15/ReadVariableOp_12
Csequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_16/ReadVariableOp2sequential_2/batch_normalization_16/ReadVariableOp2l
4sequential_2/batch_normalization_16/ReadVariableOp_14sequential_2/batch_normalization_16/ReadVariableOp_12
Csequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_17/ReadVariableOp2sequential_2/batch_normalization_17/ReadVariableOp2l
4sequential_2/batch_normalization_17/ReadVariableOp_14sequential_2/batch_normalization_17/ReadVariableOp_12
Csequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_18/ReadVariableOp2sequential_2/batch_normalization_18/ReadVariableOp2l
4sequential_2/batch_normalization_18/ReadVariableOp_14sequential_2/batch_normalization_18/ReadVariableOp_12
Csequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_19/ReadVariableOp2sequential_2/batch_normalization_19/ReadVariableOp2l
4sequential_2/batch_normalization_19/ReadVariableOp_14sequential_2/batch_normalization_19/ReadVariableOp_12|
<sequential_2/batch_normalization_20/batchnorm/ReadVariableOp<sequential_2/batch_normalization_20/batchnorm/ReadVariableOp2
>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_1>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_12
>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_2>sequential_2/batch_normalization_20/batchnorm/ReadVariableOp_22
@sequential_2/batch_normalization_20/batchnorm/mul/ReadVariableOp@sequential_2/batch_normalization_20/batchnorm/mul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp2r
7sequential_2/separable_conv2d_12/BiasAdd/ReadVariableOp7sequential_2/separable_conv2d_12/BiasAdd/ReadVariableOp2
@sequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp@sequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp2
Bsequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp_1Bsequential_2/separable_conv2d_12/separable_conv2d/ReadVariableOp_12r
7sequential_2/separable_conv2d_13/BiasAdd/ReadVariableOp7sequential_2/separable_conv2d_13/BiasAdd/ReadVariableOp2
@sequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp@sequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp2
Bsequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp_1Bsequential_2/separable_conv2d_13/separable_conv2d/ReadVariableOp_12r
7sequential_2/separable_conv2d_14/BiasAdd/ReadVariableOp7sequential_2/separable_conv2d_14/BiasAdd/ReadVariableOp2
@sequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp@sequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp2
Bsequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_1Bsequential_2/separable_conv2d_14/separable_conv2d/ReadVariableOp_12r
7sequential_2/separable_conv2d_15/BiasAdd/ReadVariableOp7sequential_2/separable_conv2d_15/BiasAdd/ReadVariableOp2
@sequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp@sequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp2
Bsequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_1Bsequential_2/separable_conv2d_15/separable_conv2d/ReadVariableOp_12r
7sequential_2/separable_conv2d_16/BiasAdd/ReadVariableOp7sequential_2/separable_conv2d_16/BiasAdd/ReadVariableOp2
@sequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp@sequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp2
Bsequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_1Bsequential_2/separable_conv2d_16/separable_conv2d/ReadVariableOp_12r
7sequential_2/separable_conv2d_17/BiasAdd/ReadVariableOp7sequential_2/separable_conv2d_17/BiasAdd/ReadVariableOp2
@sequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp@sequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp2
Bsequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1Bsequential_2/separable_conv2d_17/separable_conv2d/ReadVariableOp_1:j f
/
_output_shapes
:’’’’’’’’’00
3
_user_specified_nameseparable_conv2d_12_input
é
J
.__inference_activation_15_layer_call_fn_323027

inputs
identityŅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_3205842
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
÷
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_320838

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä

R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323138

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ś
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ų”
¶
H__inference_sequential_2_layer_call_and_return_conditional_losses_321626

inputs4
separable_conv2d_12_321495:4
separable_conv2d_12_321497: (
separable_conv2d_12_321499: +
batch_normalization_14_321503: +
batch_normalization_14_321505: +
batch_normalization_14_321507: +
batch_normalization_14_321509: 4
separable_conv2d_13_321514: 4
separable_conv2d_13_321516: @(
separable_conv2d_13_321518:@+
batch_normalization_15_321522:@+
batch_normalization_15_321524:@+
batch_normalization_15_321526:@+
batch_normalization_15_321528:@4
separable_conv2d_14_321531:@4
separable_conv2d_14_321533:@@(
separable_conv2d_14_321535:@+
batch_normalization_16_321539:@+
batch_normalization_16_321541:@+
batch_normalization_16_321543:@+
batch_normalization_16_321545:@4
separable_conv2d_15_321550:@5
separable_conv2d_15_321552:@)
separable_conv2d_15_321554:	,
batch_normalization_17_321558:	,
batch_normalization_17_321560:	,
batch_normalization_17_321562:	,
batch_normalization_17_321564:	5
separable_conv2d_16_321567:6
separable_conv2d_16_321569:)
separable_conv2d_16_321571:	,
batch_normalization_18_321575:	,
batch_normalization_18_321577:	,
batch_normalization_18_321579:	,
batch_normalization_18_321581:	5
separable_conv2d_17_321584:6
separable_conv2d_17_321586:)
separable_conv2d_17_321588:	,
batch_normalization_19_321592:	,
batch_normalization_19_321594:	,
batch_normalization_19_321596:	,
batch_normalization_19_321598:	"
dense_4_321604:
$
dense_4_321606:	,
batch_normalization_20_321610:	,
batch_normalization_20_321612:	,
batch_normalization_20_321614:	,
batch_normalization_20_321616:	!
dense_5_321620:	
dense_5_321622:
identity¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢"dropout_10/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢+separable_conv2d_12/StatefulPartitionedCall¢+separable_conv2d_13/StatefulPartitionedCall¢+separable_conv2d_14/StatefulPartitionedCall¢+separable_conv2d_15/StatefulPartitionedCall¢+separable_conv2d_16/StatefulPartitionedCall¢+separable_conv2d_17/StatefulPartitionedCallō
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsseparable_conv2d_12_321495separable_conv2d_12_321497separable_conv2d_12_321499*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_3194102-
+separable_conv2d_12/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_3205352
activation_14/PartitionedCallÅ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0batch_normalization_14_321503batch_normalization_14_321505batch_normalization_14_321507batch_normalization_14_321509*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_32136820
.batch_normalization_14/StatefulPartitionedCall„
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3195542!
max_pooling2d_6/PartitionedCall
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3213322#
!dropout_8/StatefulPartitionedCall
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0separable_conv2d_13_321514separable_conv2d_13_321516separable_conv2d_13_321518*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_3195762-
+separable_conv2d_13/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_3205842
activation_15/PartitionedCallÅ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0batch_normalization_15_321522batch_normalization_15_321524batch_normalization_15_321526batch_normalization_15_321528*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_32129520
.batch_normalization_15/StatefulPartitionedCall„
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0separable_conv2d_14_321531separable_conv2d_14_321533separable_conv2d_14_321535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_3197302-
+separable_conv2d_14/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_3206252
activation_16/PartitionedCallÅ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_16_321539batch_normalization_16_321541batch_normalization_16_321543batch_normalization_16_321545*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_32124520
.batch_normalization_16/StatefulPartitionedCall„
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3198742!
max_pooling2d_7/PartitionedCallĄ
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3212092#
!dropout_9/StatefulPartitionedCall
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0separable_conv2d_15_321550separable_conv2d_15_321552separable_conv2d_15_321554*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_3198962-
+separable_conv2d_15/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_3206742
activation_17/PartitionedCallĘ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_17_321558batch_normalization_17_321560batch_normalization_17_321562batch_normalization_17_321564*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_32117220
.batch_normalization_17/StatefulPartitionedCall¦
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0separable_conv2d_16_321567separable_conv2d_16_321569separable_conv2d_16_321571*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_3200502-
+separable_conv2d_16/StatefulPartitionedCall
activation_18/PartitionedCallPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_3207152
activation_18/PartitionedCallĘ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_18_321575batch_normalization_18_321577batch_normalization_18_321579batch_normalization_18_321581*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_32112220
.batch_normalization_18/StatefulPartitionedCall¦
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0separable_conv2d_17_321584separable_conv2d_17_321586separable_conv2d_17_321588*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_3202042-
+separable_conv2d_17/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_3207562
activation_19/PartitionedCallĘ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0batch_normalization_19_321592batch_normalization_19_321594batch_normalization_19_321596batch_normalization_19_321598*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_32107220
.batch_normalization_19/StatefulPartitionedCall¦
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3203482!
max_pooling2d_8/PartitionedCallÄ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3210362$
"dropout_10/StatefulPartitionedCall
flatten_2/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3207992
flatten_2/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_321604dense_4_321606*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3208112!
dense_4/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_3208222
activation_20/PartitionedCall¾
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0batch_normalization_20_321610batch_normalization_20_321612batch_normalization_20_321614batch_normalization_20_321616*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_32043820
.batch_normalization_20/StatefulPartitionedCallĢ
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3209912$
"dropout_11/StatefulPartitionedCall·
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_321620dense_5_321622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3208512!
dense_5/StatefulPartitionedCall½
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
į
F
*__inference_dropout_8_layer_call_fn_323000

inputs
identityĪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3205702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’ :W S
/
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

c
E__inference_dropout_9_layer_call_and_return_conditional_losses_323305

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ķ
J
.__inference_activation_17_layer_call_fn_323322

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_3206742
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

e
I__inference_activation_17_layer_call_and_return_conditional_losses_320674

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦
Ņ
7__inference_batch_normalization_16_layer_call_fn_323205

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3206442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ń
¤
H__inference_sequential_2_layer_call_and_return_conditional_losses_320858

inputs4
separable_conv2d_12_320523:4
separable_conv2d_12_320525: (
separable_conv2d_12_320527: +
batch_normalization_14_320555: +
batch_normalization_14_320557: +
batch_normalization_14_320559: +
batch_normalization_14_320561: 4
separable_conv2d_13_320572: 4
separable_conv2d_13_320574: @(
separable_conv2d_13_320576:@+
batch_normalization_15_320604:@+
batch_normalization_15_320606:@+
batch_normalization_15_320608:@+
batch_normalization_15_320610:@4
separable_conv2d_14_320613:@4
separable_conv2d_14_320615:@@(
separable_conv2d_14_320617:@+
batch_normalization_16_320645:@+
batch_normalization_16_320647:@+
batch_normalization_16_320649:@+
batch_normalization_16_320651:@4
separable_conv2d_15_320662:@5
separable_conv2d_15_320664:@)
separable_conv2d_15_320666:	,
batch_normalization_17_320694:	,
batch_normalization_17_320696:	,
batch_normalization_17_320698:	,
batch_normalization_17_320700:	5
separable_conv2d_16_320703:6
separable_conv2d_16_320705:)
separable_conv2d_16_320707:	,
batch_normalization_18_320735:	,
batch_normalization_18_320737:	,
batch_normalization_18_320739:	,
batch_normalization_18_320741:	5
separable_conv2d_17_320744:6
separable_conv2d_17_320746:)
separable_conv2d_17_320748:	,
batch_normalization_19_320776:	,
batch_normalization_19_320778:	,
batch_normalization_19_320780:	,
batch_normalization_19_320782:	"
dense_4_320812:
$
dense_4_320814:	,
batch_normalization_20_320824:	,
batch_normalization_20_320826:	,
batch_normalization_20_320828:	,
batch_normalization_20_320830:	!
dense_5_320852:	
dense_5_320854:
identity¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢+separable_conv2d_12/StatefulPartitionedCall¢+separable_conv2d_13/StatefulPartitionedCall¢+separable_conv2d_14/StatefulPartitionedCall¢+separable_conv2d_15/StatefulPartitionedCall¢+separable_conv2d_16/StatefulPartitionedCall¢+separable_conv2d_17/StatefulPartitionedCallō
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsseparable_conv2d_12_320523separable_conv2d_12_320525separable_conv2d_12_320527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_3194102-
+separable_conv2d_12/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_3205352
activation_14/PartitionedCallĒ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0batch_normalization_14_320555batch_normalization_14_320557batch_normalization_14_320559batch_normalization_14_320561*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_32055420
.batch_normalization_14/StatefulPartitionedCall„
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3195542!
max_pooling2d_6/PartitionedCall
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3205702
dropout_8/PartitionedCall
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0separable_conv2d_13_320572separable_conv2d_13_320574separable_conv2d_13_320576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_3195762-
+separable_conv2d_13/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_3205842
activation_15/PartitionedCallĒ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0batch_normalization_15_320604batch_normalization_15_320606batch_normalization_15_320608batch_normalization_15_320610*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_32060320
.batch_normalization_15/StatefulPartitionedCall„
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0separable_conv2d_14_320613separable_conv2d_14_320615separable_conv2d_14_320617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_3197302-
+separable_conv2d_14/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_3206252
activation_16/PartitionedCallĒ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_16_320645batch_normalization_16_320647batch_normalization_16_320649batch_normalization_16_320651*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_32064420
.batch_normalization_16/StatefulPartitionedCall„
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3198742!
max_pooling2d_7/PartitionedCall
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3206602
dropout_9/PartitionedCall
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0separable_conv2d_15_320662separable_conv2d_15_320664separable_conv2d_15_320666*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_3198962-
+separable_conv2d_15/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_3206742
activation_17/PartitionedCallČ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_17_320694batch_normalization_17_320696batch_normalization_17_320698batch_normalization_17_320700*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_32069320
.batch_normalization_17/StatefulPartitionedCall¦
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0separable_conv2d_16_320703separable_conv2d_16_320705separable_conv2d_16_320707*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_3200502-
+separable_conv2d_16/StatefulPartitionedCall
activation_18/PartitionedCallPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_3207152
activation_18/PartitionedCallČ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_18_320735batch_normalization_18_320737batch_normalization_18_320739batch_normalization_18_320741*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_32073420
.batch_normalization_18/StatefulPartitionedCall¦
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0separable_conv2d_17_320744separable_conv2d_17_320746separable_conv2d_17_320748*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_3202042-
+separable_conv2d_17/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_3207562
activation_19/PartitionedCallČ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0batch_normalization_19_320776batch_normalization_19_320778batch_normalization_19_320780batch_normalization_19_320782*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_32077520
.batch_normalization_19/StatefulPartitionedCall¦
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3203482!
max_pooling2d_8/PartitionedCall
dropout_10/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3207912
dropout_10/PartitionedCallų
flatten_2/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3207992
flatten_2/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_320812dense_4_320814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3208112!
dense_4/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_3208222
activation_20/PartitionedCallĄ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0batch_normalization_20_320824batch_normalization_20_320826batch_normalization_20_320828batch_normalization_20_320830*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_32037820
.batch_normalization_20/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3208382
dropout_11/PartitionedCallÆ
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_320852dense_5_320854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3208512!
dense_5/StatefulPartitionedCall«
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
Ś

O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_319576

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
: @*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ *
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2	
BiasAddŻ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs
Š
Å
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_319974

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ķ
J
.__inference_activation_18_layer_call_fn_323456

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_3207152
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¬
Ö
7__inference_batch_normalization_17_layer_call_fn_323379

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3211722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±

õ
C__inference_dense_5_layer_call_and_return_conditional_losses_320851

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ą
Į
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323120

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
ó
d
+__inference_dropout_10_layer_call_fn_323729

inputs
identity¢StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3210362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Į
­
$__inference_signature_wrapper_322211
separable_conv2d_12_input!
unknown:#
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@$

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@%

unknown_21:@

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	%

unknown_27:&

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:&

unknown_35:

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	

unknown_41:
$

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:
identity¢StatefulPartitionedCallž
StatefulPartitionedCallStatefulPartitionedCallseparable_conv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_3193942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
/
_output_shapes
:’’’’’’’’’00
3
_user_specified_nameseparable_conv2d_12_input
Ą
Į
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_319488

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs
¬
Ö
7__inference_batch_normalization_18_layer_call_fn_323513

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_3211222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ķ
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_321209

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yĘ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs


R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_319444

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3ģ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs

”
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323531

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ķ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Å
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323585

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1’
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ō
”
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323433

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ū
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

”
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323397

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ķ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ń
¶
-__inference_sequential_2_layer_call_fn_320961
separable_conv2d_12_input!
unknown:#
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@$

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@%

unknown_21:@

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	%

unknown_27:&

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:&

unknown_35:

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	

unknown_41:
$

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallseparable_conv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3208582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
/
_output_shapes
:’’’’’’’’’00
3
_user_specified_nameseparable_conv2d_12_input
Ä

R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323272

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ś
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

d
F__inference_dropout_10_layer_call_and_return_conditional_losses_323734

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:’’’’’’’’’2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ą
Į
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_319654

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
Ō
”
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_320734

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ū
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ś	
÷
C__inference_dense_4_layer_call_and_return_conditional_losses_320811

inputs2
matmul_readvariableop_resource:
$.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
$*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’$
 
_user_specified_nameinputs
ģ
Ņ
7__inference_batch_normalization_16_layer_call_fn_323192

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3198082
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

e
I__inference_activation_15_layer_call_and_return_conditional_losses_320584

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
č

O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_320204

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1“
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp»
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate÷
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingSAME*
strides
2
separable_conv2d/depthwiseō
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp„
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2	
BiasAddŽ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
°
ķ5
H__inference_sequential_2_layer_call_and_return_conditional_losses_322861

inputsV
<separable_conv2d_12_separable_conv2d_readvariableop_resource:X
>separable_conv2d_12_separable_conv2d_readvariableop_1_resource: A
3separable_conv2d_12_biasadd_readvariableop_resource: <
.batch_normalization_14_readvariableop_resource: >
0batch_normalization_14_readvariableop_1_resource: M
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource: V
<separable_conv2d_13_separable_conv2d_readvariableop_resource: X
>separable_conv2d_13_separable_conv2d_readvariableop_1_resource: @A
3separable_conv2d_13_biasadd_readvariableop_resource:@<
.batch_normalization_15_readvariableop_resource:@>
0batch_normalization_15_readvariableop_1_resource:@M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource:@V
<separable_conv2d_14_separable_conv2d_readvariableop_resource:@X
>separable_conv2d_14_separable_conv2d_readvariableop_1_resource:@@A
3separable_conv2d_14_biasadd_readvariableop_resource:@<
.batch_normalization_16_readvariableop_resource:@>
0batch_normalization_16_readvariableop_1_resource:@M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@V
<separable_conv2d_15_separable_conv2d_readvariableop_resource:@Y
>separable_conv2d_15_separable_conv2d_readvariableop_1_resource:@B
3separable_conv2d_15_biasadd_readvariableop_resource:	=
.batch_normalization_17_readvariableop_resource:	?
0batch_normalization_17_readvariableop_1_resource:	N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	W
<separable_conv2d_16_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_16_separable_conv2d_readvariableop_1_resource:B
3separable_conv2d_16_biasadd_readvariableop_resource:	=
.batch_normalization_18_readvariableop_resource:	?
0batch_normalization_18_readvariableop_1_resource:	N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	W
<separable_conv2d_17_separable_conv2d_readvariableop_resource:Z
>separable_conv2d_17_separable_conv2d_readvariableop_1_resource:B
3separable_conv2d_17_biasadd_readvariableop_resource:	=
.batch_normalization_19_readvariableop_resource:	?
0batch_normalization_19_readvariableop_1_resource:	N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	:
&dense_4_matmul_readvariableop_resource:
$6
'dense_4_biasadd_readvariableop_resource:	M
>batch_normalization_20_assignmovingavg_readvariableop_resource:	O
@batch_normalization_20_assignmovingavg_1_readvariableop_resource:	K
<batch_normalization_20_batchnorm_mul_readvariableop_resource:	G
8batch_normalization_20_batchnorm_readvariableop_resource:	9
&dense_5_matmul_readvariableop_resource:	5
'dense_5_biasadd_readvariableop_resource:
identity¢%batch_normalization_14/AssignNewValue¢'batch_normalization_14/AssignNewValue_1¢6batch_normalization_14/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_14/ReadVariableOp¢'batch_normalization_14/ReadVariableOp_1¢%batch_normalization_15/AssignNewValue¢'batch_normalization_15/AssignNewValue_1¢6batch_normalization_15/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_15/ReadVariableOp¢'batch_normalization_15/ReadVariableOp_1¢%batch_normalization_16/AssignNewValue¢'batch_normalization_16/AssignNewValue_1¢6batch_normalization_16/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_16/ReadVariableOp¢'batch_normalization_16/ReadVariableOp_1¢%batch_normalization_17/AssignNewValue¢'batch_normalization_17/AssignNewValue_1¢6batch_normalization_17/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_17/ReadVariableOp¢'batch_normalization_17/ReadVariableOp_1¢%batch_normalization_18/AssignNewValue¢'batch_normalization_18/AssignNewValue_1¢6batch_normalization_18/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_18/ReadVariableOp¢'batch_normalization_18/ReadVariableOp_1¢%batch_normalization_19/AssignNewValue¢'batch_normalization_19/AssignNewValue_1¢6batch_normalization_19/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_19/ReadVariableOp¢'batch_normalization_19/ReadVariableOp_1¢&batch_normalization_20/AssignMovingAvg¢5batch_normalization_20/AssignMovingAvg/ReadVariableOp¢(batch_normalization_20/AssignMovingAvg_1¢7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_20/batchnorm/ReadVariableOp¢3batch_normalization_20/batchnorm/mul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢*separable_conv2d_12/BiasAdd/ReadVariableOp¢3separable_conv2d_12/separable_conv2d/ReadVariableOp¢5separable_conv2d_12/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_13/BiasAdd/ReadVariableOp¢3separable_conv2d_13/separable_conv2d/ReadVariableOp¢5separable_conv2d_13/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_14/BiasAdd/ReadVariableOp¢3separable_conv2d_14/separable_conv2d/ReadVariableOp¢5separable_conv2d_14/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_15/BiasAdd/ReadVariableOp¢3separable_conv2d_15/separable_conv2d/ReadVariableOp¢5separable_conv2d_15/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_16/BiasAdd/ReadVariableOp¢3separable_conv2d_16/separable_conv2d/ReadVariableOp¢5separable_conv2d_16/separable_conv2d/ReadVariableOp_1¢*separable_conv2d_17/BiasAdd/ReadVariableOp¢3separable_conv2d_17/separable_conv2d/ReadVariableOp¢5separable_conv2d_17/separable_conv2d/ReadVariableOp_1ļ
3separable_conv2d_12/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_12_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3separable_conv2d_12/separable_conv2d/ReadVariableOpõ
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_12_separable_conv2d_readvariableop_1_resource*&
_output_shapes
: *
dtype027
5separable_conv2d_12/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_12/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_12/separable_conv2d/Shape¹
2separable_conv2d_12/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_12/separable_conv2d/dilation_rate 
.separable_conv2d_12/separable_conv2d/depthwiseDepthwiseConv2dNativeinputs;separable_conv2d_12/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’00*
paddingSAME*
strides
20
.separable_conv2d_12/separable_conv2d/depthwise±
$separable_conv2d_12/separable_conv2dConv2D7separable_conv2d_12/separable_conv2d/depthwise:output:0=separable_conv2d_12/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’00 *
paddingVALID*
strides
2&
$separable_conv2d_12/separable_conv2dČ
*separable_conv2d_12/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*separable_conv2d_12/BiasAdd/ReadVariableOpā
separable_conv2d_12/BiasAddBiasAdd-separable_conv2d_12/separable_conv2d:output:02separable_conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’00 2
separable_conv2d_12/BiasAdd
activation_14/ReluRelu$separable_conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’00 2
activation_14/Relu¹
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_14/ReadVariableOpæ
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_14/ReadVariableOp_1ģ
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpņ
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ü
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3 activation_14/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_14/FusedBatchNormV3µ
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_14/AssignNewValueĮ
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_14/AssignNewValue_1×
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_14/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPoolw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout_8/dropout/Const³
dropout_8/dropout/MulMul max_pooling2d_6/MaxPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout_8/dropout/Mul
dropout_8/dropout/ShapeShape max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/ShapeŚ
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’ *
dtype020
.dropout_8/dropout/random_uniform/RandomUniform
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2"
 dropout_8/dropout/GreaterEqual/yī
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2 
dropout_8/dropout/GreaterEqual„
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’ 2
dropout_8/dropout/CastŖ
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’ 2
dropout_8/dropout/Mul_1ļ
3separable_conv2d_13/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_13_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype025
3separable_conv2d_13/separable_conv2d/ReadVariableOpõ
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_13_separable_conv2d_readvariableop_1_resource*&
_output_shapes
: @*
dtype027
5separable_conv2d_13/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_13/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*separable_conv2d_13/separable_conv2d/Shape¹
2separable_conv2d_13/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_13/separable_conv2d/dilation_rateµ
.separable_conv2d_13/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_8/dropout/Mul_1:z:0;separable_conv2d_13/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’ *
paddingSAME*
strides
20
.separable_conv2d_13/separable_conv2d/depthwise±
$separable_conv2d_13/separable_conv2dConv2D7separable_conv2d_13/separable_conv2d/depthwise:output:0=separable_conv2d_13/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
2&
$separable_conv2d_13/separable_conv2dČ
*separable_conv2d_13/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*separable_conv2d_13/BiasAdd/ReadVariableOpā
separable_conv2d_13/BiasAddBiasAdd-separable_conv2d_13/separable_conv2d:output:02separable_conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@2
separable_conv2d_13/BiasAdd
activation_15/ReluRelu$separable_conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
activation_15/Relu¹
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOpæ
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1ģ
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpņ
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ü
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3 activation_15/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_15/FusedBatchNormV3µ
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValueĮ
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1ļ
3separable_conv2d_14/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_14_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype025
3separable_conv2d_14/separable_conv2d/ReadVariableOpõ
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_14_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@@*
dtype027
5separable_conv2d_14/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_14/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2,
*separable_conv2d_14/separable_conv2d/Shape¹
2separable_conv2d_14/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_14/separable_conv2d/dilation_rateÅ
.separable_conv2d_14/separable_conv2d/depthwiseDepthwiseConv2dNative+batch_normalization_15/FusedBatchNormV3:y:0;separable_conv2d_14/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
20
.separable_conv2d_14/separable_conv2d/depthwise±
$separable_conv2d_14/separable_conv2dConv2D7separable_conv2d_14/separable_conv2d/depthwise:output:0=separable_conv2d_14/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
2&
$separable_conv2d_14/separable_conv2dČ
*separable_conv2d_14/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*separable_conv2d_14/BiasAdd/ReadVariableOpā
separable_conv2d_14/BiasAddBiasAdd-separable_conv2d_14/separable_conv2d:output:02separable_conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@2
separable_conv2d_14/BiasAdd
activation_16/ReluRelu$separable_conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
activation_16/Relu¹
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOpæ
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1ģ
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpņ
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ü
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 activation_16/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_16/FusedBatchNormV3µ
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValueĮ
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1×
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:’’’’’’’’’@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPoolw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout_9/dropout/Const³
dropout_9/dropout/MulMul max_pooling2d_7/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout_9/dropout/Mul
dropout_9/dropout/ShapeShape max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/ShapeŚ
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’@*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2"
 dropout_9/dropout/GreaterEqual/yī
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2 
dropout_9/dropout/GreaterEqual„
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’@2
dropout_9/dropout/CastŖ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout_9/dropout/Mul_1ļ
3separable_conv2d_15/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_15_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype025
3separable_conv2d_15/separable_conv2d/ReadVariableOpö
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_15_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@*
dtype027
5separable_conv2d_15/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_15/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2,
*separable_conv2d_15/separable_conv2d/Shape¹
2separable_conv2d_15/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_15/separable_conv2d/dilation_rateµ
.separable_conv2d_15/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_9/dropout/Mul_1:z:0;separable_conv2d_15/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’@*
paddingSAME*
strides
20
.separable_conv2d_15/separable_conv2d/depthwise²
$separable_conv2d_15/separable_conv2dConv2D7separable_conv2d_15/separable_conv2d/depthwise:output:0=separable_conv2d_15/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2&
$separable_conv2d_15/separable_conv2dÉ
*separable_conv2d_15/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*separable_conv2d_15/BiasAdd/ReadVariableOpć
separable_conv2d_15/BiasAddBiasAdd-separable_conv2d_15/separable_conv2d:output:02separable_conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
separable_conv2d_15/BiasAdd
activation_17/ReluRelu$separable_conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
activation_17/Reluŗ
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_17/ReadVariableOpĄ
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_17/ReadVariableOp_1ķ
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 activation_17/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_17/FusedBatchNormV3µ
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValueĮ
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1š
3separable_conv2d_16/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_16_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype025
3separable_conv2d_16/separable_conv2d/ReadVariableOp÷
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_16_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype027
5separable_conv2d_16/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_16/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_16/separable_conv2d/Shape¹
2separable_conv2d_16/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_16/separable_conv2d/dilation_rateĘ
.separable_conv2d_16/separable_conv2d/depthwiseDepthwiseConv2dNative+batch_normalization_17/FusedBatchNormV3:y:0;separable_conv2d_16/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
20
.separable_conv2d_16/separable_conv2d/depthwise²
$separable_conv2d_16/separable_conv2dConv2D7separable_conv2d_16/separable_conv2d/depthwise:output:0=separable_conv2d_16/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2&
$separable_conv2d_16/separable_conv2dÉ
*separable_conv2d_16/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*separable_conv2d_16/BiasAdd/ReadVariableOpć
separable_conv2d_16/BiasAddBiasAdd-separable_conv2d_16/separable_conv2d:output:02separable_conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
separable_conv2d_16/BiasAdd
activation_18/ReluRelu$separable_conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
activation_18/Reluŗ
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_18/ReadVariableOpĄ
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_18/ReadVariableOp_1ķ
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 activation_18/Relu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_18/FusedBatchNormV3µ
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_18/AssignNewValueĮ
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_18/AssignNewValue_1š
3separable_conv2d_17/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_17_separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype025
3separable_conv2d_17/separable_conv2d/ReadVariableOp÷
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_17_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype027
5separable_conv2d_17/separable_conv2d/ReadVariableOp_1±
*separable_conv2d_17/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_17/separable_conv2d/Shape¹
2separable_conv2d_17/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_17/separable_conv2d/dilation_rateĘ
.separable_conv2d_17/separable_conv2d/depthwiseDepthwiseConv2dNative+batch_normalization_18/FusedBatchNormV3:y:0;separable_conv2d_17/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingSAME*
strides
20
.separable_conv2d_17/separable_conv2d/depthwise²
$separable_conv2d_17/separable_conv2dConv2D7separable_conv2d_17/separable_conv2d/depthwise:output:0=separable_conv2d_17/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2&
$separable_conv2d_17/separable_conv2dÉ
*separable_conv2d_17/BiasAdd/ReadVariableOpReadVariableOp3separable_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*separable_conv2d_17/BiasAdd/ReadVariableOpć
separable_conv2d_17/BiasAddBiasAdd-separable_conv2d_17/separable_conv2d:output:02separable_conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:’’’’’’’’’2
separable_conv2d_17/BiasAdd
activation_19/ReluRelu$separable_conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
activation_19/Reluŗ
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_19/ReadVariableOpĄ
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_19/ReadVariableOp_1ķ
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3 activation_19/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_19/FusedBatchNormV3µ
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_19/AssignNewValueĮ
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_19/AssignNewValue_1Ų
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_19/FusedBatchNormV3:y:0*0
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPooly
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout_10/dropout/Const·
dropout_10/dropout/MulMul max_pooling2d_8/MaxPool:output:0!dropout_10/dropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_10/dropout/Mul
dropout_10/dropout/ShapeShape max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_10/dropout/ShapeŽ
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_10/dropout/GreaterEqual/yó
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2!
dropout_10/dropout/GreaterEqual©
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout_10/dropout/CastÆ
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout_10/dropout/Mul_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
flatten_2/Const
flatten_2/ReshapeReshapedropout_10/dropout/Mul_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2
flatten_2/Reshape§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
$*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulflatten_2/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/MatMul„
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
dense_4/BiasAdd}
activation_20/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation_20/Reluø
5batch_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_20/moments/mean/reduction_indicesļ
#batch_normalization_20/moments/meanMean activation_20/Relu:activations:0>batch_normalization_20/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2%
#batch_normalization_20/moments/meanĀ
+batch_normalization_20/moments/StopGradientStopGradient,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes
:	2-
+batch_normalization_20/moments/StopGradient
0batch_normalization_20/moments/SquaredDifferenceSquaredDifference activation_20/Relu:activations:04batch_normalization_20/moments/StopGradient:output:0*
T0*(
_output_shapes
:’’’’’’’’’22
0batch_normalization_20/moments/SquaredDifferenceĄ
9batch_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_20/moments/variance/reduction_indices
'batch_normalization_20/moments/varianceMean4batch_normalization_20/moments/SquaredDifference:z:0Bbatch_normalization_20/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2)
'batch_normalization_20/moments/varianceĘ
&batch_normalization_20/moments/SqueezeSqueeze,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2(
&batch_normalization_20/moments/SqueezeĪ
(batch_normalization_20/moments/Squeeze_1Squeeze0batch_normalization_20/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2*
(batch_normalization_20/moments/Squeeze_1”
,batch_normalization_20/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_20/AssignMovingAvg/decayź
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_20/AssignMovingAvg/ReadVariableOpõ
*batch_normalization_20/AssignMovingAvg/subSub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_20/moments/Squeeze:output:0*
T0*
_output_shapes	
:2,
*batch_normalization_20/AssignMovingAvg/subģ
*batch_normalization_20/AssignMovingAvg/mulMul.batch_normalization_20/AssignMovingAvg/sub:z:05batch_normalization_20/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2,
*batch_normalization_20/AssignMovingAvg/mul²
&batch_normalization_20/AssignMovingAvgAssignSubVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_20/AssignMovingAvg„
.batch_normalization_20/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_20/AssignMovingAvg_1/decayš
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype029
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpż
,batch_normalization_20/AssignMovingAvg_1/subSub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_20/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2.
,batch_normalization_20/AssignMovingAvg_1/subō
,batch_normalization_20/AssignMovingAvg_1/mulMul0batch_normalization_20/AssignMovingAvg_1/sub:z:07batch_normalization_20/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2.
,batch_normalization_20/AssignMovingAvg_1/mul¼
(batch_normalization_20/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource0batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_20/AssignMovingAvg_1
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_20/batchnorm/add/yß
$batch_normalization_20/batchnorm/addAddV21batch_normalization_20/moments/Squeeze_1:output:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_20/batchnorm/add©
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_20/batchnorm/Rsqrtä
3batch_normalization_20/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_20_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_20/batchnorm/mul/ReadVariableOpā
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:0;batch_normalization_20/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_20/batchnorm/mulÖ
&batch_normalization_20/batchnorm/mul_1Mul activation_20/Relu:activations:0(batch_normalization_20/batchnorm/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2(
&batch_normalization_20/batchnorm/mul_1Ų
&batch_normalization_20/batchnorm/mul_2Mul/batch_normalization_20/moments/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_20/batchnorm/mul_2Ų
/batch_normalization_20/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_20_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_20/batchnorm/ReadVariableOpŽ
$batch_normalization_20/batchnorm/subSub7batch_normalization_20/batchnorm/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_20/batchnorm/subā
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2(
&batch_normalization_20/batchnorm/add_1y
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Const¹
dropout_11/dropout/MulMul*batch_normalization_20/batchnorm/add_1:z:0!dropout_11/dropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShape*batch_normalization_20/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_11/dropout/ShapeÖ
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/yė
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2!
dropout_11/dropout/GreaterEqual”
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout_11/dropout/Cast§
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_11/dropout/Mul_1¦
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_5/MatMul/ReadVariableOp”
dense_5/MatMulMatMuldropout_11/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp”
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_5/SigmoidÉ
IdentityIdentitydense_5/Sigmoid:y:0&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1'^batch_normalization_20/AssignMovingAvg6^batch_normalization_20/AssignMovingAvg/ReadVariableOp)^batch_normalization_20/AssignMovingAvg_18^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_20/batchnorm/ReadVariableOp4^batch_normalization_20/batchnorm/mul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp+^separable_conv2d_12/BiasAdd/ReadVariableOp4^separable_conv2d_12/separable_conv2d/ReadVariableOp6^separable_conv2d_12/separable_conv2d/ReadVariableOp_1+^separable_conv2d_13/BiasAdd/ReadVariableOp4^separable_conv2d_13/separable_conv2d/ReadVariableOp6^separable_conv2d_13/separable_conv2d/ReadVariableOp_1+^separable_conv2d_14/BiasAdd/ReadVariableOp4^separable_conv2d_14/separable_conv2d/ReadVariableOp6^separable_conv2d_14/separable_conv2d/ReadVariableOp_1+^separable_conv2d_15/BiasAdd/ReadVariableOp4^separable_conv2d_15/separable_conv2d/ReadVariableOp6^separable_conv2d_15/separable_conv2d/ReadVariableOp_1+^separable_conv2d_16/BiasAdd/ReadVariableOp4^separable_conv2d_16/separable_conv2d/ReadVariableOp6^separable_conv2d_16/separable_conv2d/ReadVariableOp_1+^separable_conv2d_17/BiasAdd/ReadVariableOp4^separable_conv2d_17/separable_conv2d/ReadVariableOp6^separable_conv2d_17/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12P
&batch_normalization_20/AssignMovingAvg&batch_normalization_20/AssignMovingAvg2n
5batch_normalization_20/AssignMovingAvg/ReadVariableOp5batch_normalization_20/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_20/AssignMovingAvg_1(batch_normalization_20/AssignMovingAvg_12r
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_20/batchnorm/ReadVariableOp/batch_normalization_20/batchnorm/ReadVariableOp2j
3batch_normalization_20/batchnorm/mul/ReadVariableOp3batch_normalization_20/batchnorm/mul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2X
*separable_conv2d_12/BiasAdd/ReadVariableOp*separable_conv2d_12/BiasAdd/ReadVariableOp2j
3separable_conv2d_12/separable_conv2d/ReadVariableOp3separable_conv2d_12/separable_conv2d/ReadVariableOp2n
5separable_conv2d_12/separable_conv2d/ReadVariableOp_15separable_conv2d_12/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_13/BiasAdd/ReadVariableOp*separable_conv2d_13/BiasAdd/ReadVariableOp2j
3separable_conv2d_13/separable_conv2d/ReadVariableOp3separable_conv2d_13/separable_conv2d/ReadVariableOp2n
5separable_conv2d_13/separable_conv2d/ReadVariableOp_15separable_conv2d_13/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_14/BiasAdd/ReadVariableOp*separable_conv2d_14/BiasAdd/ReadVariableOp2j
3separable_conv2d_14/separable_conv2d/ReadVariableOp3separable_conv2d_14/separable_conv2d/ReadVariableOp2n
5separable_conv2d_14/separable_conv2d/ReadVariableOp_15separable_conv2d_14/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_15/BiasAdd/ReadVariableOp*separable_conv2d_15/BiasAdd/ReadVariableOp2j
3separable_conv2d_15/separable_conv2d/ReadVariableOp3separable_conv2d_15/separable_conv2d/ReadVariableOp2n
5separable_conv2d_15/separable_conv2d/ReadVariableOp_15separable_conv2d_15/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_16/BiasAdd/ReadVariableOp*separable_conv2d_16/BiasAdd/ReadVariableOp2j
3separable_conv2d_16/separable_conv2d/ReadVariableOp3separable_conv2d_16/separable_conv2d/ReadVariableOp2n
5separable_conv2d_16/separable_conv2d/ReadVariableOp_15separable_conv2d_16/separable_conv2d/ReadVariableOp_12X
*separable_conv2d_17/BiasAdd/ReadVariableOp*separable_conv2d_17/BiasAdd/ReadVariableOp2j
3separable_conv2d_17/separable_conv2d/ReadVariableOp3separable_conv2d_17/separable_conv2d/ReadVariableOp2n
5separable_conv2d_17/separable_conv2d/ReadVariableOp_15separable_conv2d_17/separable_conv2d/ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs

e
I__inference_activation_15_layer_call_and_return_conditional_losses_323032

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

”
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_320084

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ķ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ķ
J
.__inference_activation_19_layer_call_fn_323590

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_3207562
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ą
Ņ
4__inference_separable_conv2d_17_layer_call_fn_320216

inputs"
unknown:%
	unknown_0:
	unknown_1:	
identity¢StatefulPartitionedCallŖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_3202042
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ō
”
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323701

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ū
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
µ
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_323893

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦
Ņ
7__inference_batch_normalization_14_layer_call_fn_322910

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3205542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’00 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
é
J
.__inference_activation_16_layer_call_fn_323161

inputs
identityŅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_3206252
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ģ
Ņ
7__inference_batch_normalization_14_layer_call_fn_322897

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3194882
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs
ē
G
+__inference_dropout_10_layer_call_fn_323724

inputs
identityŠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3207912
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ö
Ö
7__inference_batch_normalization_19_layer_call_fn_323608

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3202382
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ų
Į
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_321295

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ž
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

c
E__inference_dropout_8_layer_call_and_return_conditional_losses_323010

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’ :W S
/
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
ć
¶
-__inference_sequential_2_layer_call_fn_321834
separable_conv2d_12_input!
unknown:#
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@$

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@%

unknown_21:@

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	%

unknown_27:&

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:&

unknown_35:

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	

unknown_41:
$

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallseparable_conv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*F
_read_only_resource_inputs(
&$	
 !$%&'(+,/012*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3216262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
/
_output_shapes
:’’’’’’’’’00
3
_user_specified_nameseparable_conv2d_12_input
ģ
Ņ
7__inference_batch_normalization_15_layer_call_fn_323058

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3196542
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
¤
Ņ
7__inference_batch_normalization_16_layer_call_fn_323218

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3212452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ś

O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_319410

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
: *
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ *
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2	
BiasAddŻ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
±

õ
C__inference_dense_5_layer_call_and_return_conditional_losses_323913

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
Į
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323156

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ž
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs


R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323102

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ģ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs


(__inference_dense_5_layer_call_fn_323902

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3208512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

c
E__inference_dropout_8_layer_call_and_return_conditional_losses_320570

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’ :W S
/
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

”
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323665

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ķ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Ö
7__inference_batch_normalization_20_layer_call_fn_323799

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_3203782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ö
Ö
7__inference_batch_normalization_18_layer_call_fn_323474

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_3200842
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ų
Ī
4__inference_separable_conv2d_13_layer_call_fn_319588

inputs!
unknown: #
	unknown_0: @
	unknown_1:@
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_3195762
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs
Ś	
÷
C__inference_dense_4_layer_call_and_return_conditional_losses_323776

inputs2
matmul_readvariableop_resource:
$.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
$*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’$
 
_user_specified_nameinputs
é
J
.__inference_activation_14_layer_call_fn_322866

inputs
identityŅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_3205352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’00 :W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
*
ļ
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_323866

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient„
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay„
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mulæ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp”
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä

R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_320554

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’00 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ś
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’00 : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
«
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_320348

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

e
I__inference_activation_14_layer_call_and_return_conditional_losses_322871

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’00 :W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
Ō
”
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_320693

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ū
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ
J
.__inference_activation_20_layer_call_fn_323781

inputs
identityĖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_3208222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
č

O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_320050

inputsC
(separable_conv2d_readvariableop_resource:F
*separable_conv2d_readvariableop_1_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1“
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp»
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate÷
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingSAME*
strides
2
separable_conv2d/depthwiseō
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp„
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2	
BiasAddŽ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

”
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_319930

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1į
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ķ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
®
Ö
7__inference_batch_normalization_19_layer_call_fn_323634

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3207752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±
µ
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_323832

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ō
Ö
7__inference_batch_normalization_17_layer_call_fn_323353

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3199742
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
é
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_320799

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Å
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323719

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1’
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ö
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_321036

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ų
Ī
4__inference_separable_conv2d_12_layer_call_fn_319422

inputs!
unknown:#
	unknown_0: 
	unknown_1: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_3194102
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Š
Å
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323683

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ŗ
£
-__inference_sequential_2_layer_call_fn_322421

inputs!
unknown:#
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@$

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@%

unknown_21:@

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	%

unknown_27:&

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:&

unknown_35:

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	

unknown_41:
$

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*F
_read_only_resource_inputs(
&$	
 !$%&'(+,/012*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3216262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
å
e
I__inference_activation_20_layer_call_and_return_conditional_losses_323786

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä

R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_320603

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ś
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
®
Ö
7__inference_batch_normalization_17_layer_call_fn_323366

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3206932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ī
Ņ
7__inference_batch_normalization_14_layer_call_fn_322884

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3194442
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs
Š
Å
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323415

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Å
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323451

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1’
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ö
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_323746

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yĒ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:’’’’’’’’’2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:’’’’’’’’’2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ä

R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_320644

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ś
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

e
I__inference_activation_19_layer_call_and_return_conditional_losses_320756

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
Į
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323290

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ž
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

e
I__inference_activation_16_layer_call_and_return_conditional_losses_320625

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:’’’’’’’’’@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ī
Ņ
7__inference_batch_normalization_15_layer_call_fn_323045

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3196102
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
±Ķ
į2
__inference__traced_save_324215
file_prefixC
?savev2_separable_conv2d_12_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_12_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableopC
?savev2_separable_conv2d_13_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_13_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableopC
?savev2_separable_conv2d_14_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_14_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableopC
?savev2_separable_conv2d_15_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_15_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableopC
?savev2_separable_conv2d_16_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_16_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_16_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableopC
?savev2_separable_conv2d_17_depthwise_kernel_read_readvariableopC
?savev2_separable_conv2d_17_pointwise_kernel_read_readvariableop7
3savev2_separable_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_12_depthwise_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_12_pointwise_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_separable_conv2d_12_bias_accumulator_read_readvariableopO
Ksavev2_adagrad_batch_normalization_14_gamma_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_14_beta_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_13_depthwise_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_13_pointwise_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_separable_conv2d_13_bias_accumulator_read_readvariableopO
Ksavev2_adagrad_batch_normalization_15_gamma_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_15_beta_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_14_depthwise_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_14_pointwise_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_separable_conv2d_14_bias_accumulator_read_readvariableopO
Ksavev2_adagrad_batch_normalization_16_gamma_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_16_beta_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_15_depthwise_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_15_pointwise_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_separable_conv2d_15_bias_accumulator_read_readvariableopO
Ksavev2_adagrad_batch_normalization_17_gamma_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_17_beta_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_16_depthwise_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_16_pointwise_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_separable_conv2d_16_bias_accumulator_read_readvariableopO
Ksavev2_adagrad_batch_normalization_18_gamma_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_18_beta_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_17_depthwise_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_separable_conv2d_17_pointwise_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_separable_conv2d_17_bias_accumulator_read_readvariableopO
Ksavev2_adagrad_batch_normalization_19_gamma_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_19_beta_accumulator_read_readvariableopA
=savev2_adagrad_dense_4_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_4_bias_accumulator_read_readvariableopO
Ksavev2_adagrad_batch_normalization_20_gamma_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_20_beta_accumulator_read_readvariableopA
=savev2_adagrad_dense_5_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_5_bias_accumulator_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĪ6
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*ą5
valueÖ5BÓ5^B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesĒ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*Ń
valueĒBÄ^B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices1
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_separable_conv2d_12_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_12_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop?savev2_separable_conv2d_13_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_13_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop?savev2_separable_conv2d_14_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_14_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop?savev2_separable_conv2d_15_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_15_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop?savev2_separable_conv2d_16_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_16_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_16_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop?savev2_separable_conv2d_17_depthwise_kernel_read_readvariableop?savev2_separable_conv2d_17_pointwise_kernel_read_readvariableop3savev2_separable_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopSsavev2_adagrad_separable_conv2d_12_depthwise_kernel_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_12_pointwise_kernel_accumulator_read_readvariableopGsavev2_adagrad_separable_conv2d_12_bias_accumulator_read_readvariableopKsavev2_adagrad_batch_normalization_14_gamma_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_14_beta_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_13_depthwise_kernel_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_13_pointwise_kernel_accumulator_read_readvariableopGsavev2_adagrad_separable_conv2d_13_bias_accumulator_read_readvariableopKsavev2_adagrad_batch_normalization_15_gamma_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_15_beta_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_14_depthwise_kernel_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_14_pointwise_kernel_accumulator_read_readvariableopGsavev2_adagrad_separable_conv2d_14_bias_accumulator_read_readvariableopKsavev2_adagrad_batch_normalization_16_gamma_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_16_beta_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_15_depthwise_kernel_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_15_pointwise_kernel_accumulator_read_readvariableopGsavev2_adagrad_separable_conv2d_15_bias_accumulator_read_readvariableopKsavev2_adagrad_batch_normalization_17_gamma_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_17_beta_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_16_depthwise_kernel_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_16_pointwise_kernel_accumulator_read_readvariableopGsavev2_adagrad_separable_conv2d_16_bias_accumulator_read_readvariableopKsavev2_adagrad_batch_normalization_18_gamma_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_18_beta_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_17_depthwise_kernel_accumulator_read_readvariableopSsavev2_adagrad_separable_conv2d_17_pointwise_kernel_accumulator_read_readvariableopGsavev2_adagrad_separable_conv2d_17_bias_accumulator_read_readvariableopKsavev2_adagrad_batch_normalization_19_gamma_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_19_beta_accumulator_read_readvariableop=savev2_adagrad_dense_4_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_4_bias_accumulator_read_readvariableopKsavev2_adagrad_batch_normalization_20_gamma_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_20_beta_accumulator_read_readvariableop=savev2_adagrad_dense_5_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_5_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *l
dtypesb
`2^	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesż
ś: :: : : : : : : : @:@:@:@:@:@:@:@@:@:@:@:@:@:@:@::::::::::::::::::::
$::::::	:: : : : : : : :: : : : : : @:@:@:@:@:@@:@:@:@:@:@::::::::::::::
$::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
::.*
(
_output_shapes
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::-$)
'
_output_shapes
::.%*
(
_output_shapes
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::!*

_output_shapes	
::&+"
 
_output_shapes
:
$:!,

_output_shapes	
::!-

_output_shapes	
::!.

_output_shapes	
::!/

_output_shapes	
::!0

_output_shapes	
::%1!

_output_shapes
:	: 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: :,:(
&
_output_shapes
::,;(
&
_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: : >

_output_shapes
: :,?(
&
_output_shapes
: :,@(
&
_output_shapes
: @: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:,D(
&
_output_shapes
:@:,E(
&
_output_shapes
:@@: F

_output_shapes
:@: G

_output_shapes
:@: H

_output_shapes
:@:,I(
&
_output_shapes
:@:-J)
'
_output_shapes
:@:!K

_output_shapes	
::!L

_output_shapes	
::!M

_output_shapes	
::-N)
'
_output_shapes
::.O*
(
_output_shapes
::!P

_output_shapes	
::!Q

_output_shapes	
::!R

_output_shapes	
::-S)
'
_output_shapes
::.T*
(
_output_shapes
::!U

_output_shapes	
::!V

_output_shapes	
::!W

_output_shapes	
::&X"
 
_output_shapes
:
$:!Y

_output_shapes	
::!Z

_output_shapes	
::![

_output_shapes	
::%\!

_output_shapes
:	: ]

_output_shapes
::^

_output_shapes
: 
ī
Ņ
7__inference_batch_normalization_16_layer_call_fn_323179

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3197642
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
ō
Ö
7__inference_batch_normalization_19_layer_call_fn_323621

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3202822
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ś

O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_319730

inputsB
(separable_conv2d_readvariableop_resource:@D
*separable_conv2d_readvariableop_1_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢separable_conv2d/ReadVariableOp¢!separable_conv2d/ReadVariableOp_1³
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
separable_conv2d/ReadVariableOp¹
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@@*
dtype02#
!separable_conv2d/ReadVariableOp_1
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
separable_conv2d/Shape
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rateö
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*
paddingSAME*
strides
2
separable_conv2d/depthwiseó
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*
paddingVALID*
strides
2
separable_conv2d
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2	
BiasAddŻ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs


R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_319610

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ģ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
Š
Å
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_320128

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ø
£
-__inference_sequential_2_layer_call_fn_322316

inputs!
unknown:#
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: #
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@$

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@%

unknown_21:@

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	%

unknown_27:&

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:&

unknown_35:

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	

unknown_41:
$

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_3208582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’00
 
_user_specified_nameinputs
£

(__inference_dense_4_layer_call_fn_323766

inputs
unknown:
$
	unknown_0:	
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3208112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:’’’’’’’’’$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’$
 
_user_specified_nameinputs
*
ļ
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_320438

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient„
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay„
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mulæ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp”
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ó
d
+__inference_dropout_11_layer_call_fn_323876

inputs
identity¢StatefulPartitionedCallą
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3209912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ś
L
0__inference_max_pooling2d_6_layer_call_fn_319560

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3195542
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
¤
Ņ
7__inference_batch_normalization_15_layer_call_fn_323084

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3212952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ū§
G
"__inference__traced_restore_324504
file_prefixO
5assignvariableop_separable_conv2d_12_depthwise_kernel:Q
7assignvariableop_1_separable_conv2d_12_pointwise_kernel: 9
+assignvariableop_2_separable_conv2d_12_bias: =
/assignvariableop_3_batch_normalization_14_gamma: <
.assignvariableop_4_batch_normalization_14_beta: C
5assignvariableop_5_batch_normalization_14_moving_mean: G
9assignvariableop_6_batch_normalization_14_moving_variance: Q
7assignvariableop_7_separable_conv2d_13_depthwise_kernel: Q
7assignvariableop_8_separable_conv2d_13_pointwise_kernel: @9
+assignvariableop_9_separable_conv2d_13_bias:@>
0assignvariableop_10_batch_normalization_15_gamma:@=
/assignvariableop_11_batch_normalization_15_beta:@D
6assignvariableop_12_batch_normalization_15_moving_mean:@H
:assignvariableop_13_batch_normalization_15_moving_variance:@R
8assignvariableop_14_separable_conv2d_14_depthwise_kernel:@R
8assignvariableop_15_separable_conv2d_14_pointwise_kernel:@@:
,assignvariableop_16_separable_conv2d_14_bias:@>
0assignvariableop_17_batch_normalization_16_gamma:@=
/assignvariableop_18_batch_normalization_16_beta:@D
6assignvariableop_19_batch_normalization_16_moving_mean:@H
:assignvariableop_20_batch_normalization_16_moving_variance:@R
8assignvariableop_21_separable_conv2d_15_depthwise_kernel:@S
8assignvariableop_22_separable_conv2d_15_pointwise_kernel:@;
,assignvariableop_23_separable_conv2d_15_bias:	?
0assignvariableop_24_batch_normalization_17_gamma:	>
/assignvariableop_25_batch_normalization_17_beta:	E
6assignvariableop_26_batch_normalization_17_moving_mean:	I
:assignvariableop_27_batch_normalization_17_moving_variance:	S
8assignvariableop_28_separable_conv2d_16_depthwise_kernel:T
8assignvariableop_29_separable_conv2d_16_pointwise_kernel:;
,assignvariableop_30_separable_conv2d_16_bias:	?
0assignvariableop_31_batch_normalization_18_gamma:	>
/assignvariableop_32_batch_normalization_18_beta:	E
6assignvariableop_33_batch_normalization_18_moving_mean:	I
:assignvariableop_34_batch_normalization_18_moving_variance:	S
8assignvariableop_35_separable_conv2d_17_depthwise_kernel:T
8assignvariableop_36_separable_conv2d_17_pointwise_kernel:;
,assignvariableop_37_separable_conv2d_17_bias:	?
0assignvariableop_38_batch_normalization_19_gamma:	>
/assignvariableop_39_batch_normalization_19_beta:	E
6assignvariableop_40_batch_normalization_19_moving_mean:	I
:assignvariableop_41_batch_normalization_19_moving_variance:	6
"assignvariableop_42_dense_4_kernel:
$/
 assignvariableop_43_dense_4_bias:	?
0assignvariableop_44_batch_normalization_20_gamma:	>
/assignvariableop_45_batch_normalization_20_beta:	E
6assignvariableop_46_batch_normalization_20_moving_mean:	I
:assignvariableop_47_batch_normalization_20_moving_variance:	5
"assignvariableop_48_dense_5_kernel:	.
 assignvariableop_49_dense_5_bias:*
 assignvariableop_50_adagrad_iter:	 +
!assignvariableop_51_adagrad_decay: 3
)assignvariableop_52_adagrad_learning_rate: #
assignvariableop_53_total: #
assignvariableop_54_count: %
assignvariableop_55_total_1: %
assignvariableop_56_count_1: f
Lassignvariableop_57_adagrad_separable_conv2d_12_depthwise_kernel_accumulator:f
Lassignvariableop_58_adagrad_separable_conv2d_12_pointwise_kernel_accumulator: N
@assignvariableop_59_adagrad_separable_conv2d_12_bias_accumulator: R
Dassignvariableop_60_adagrad_batch_normalization_14_gamma_accumulator: Q
Cassignvariableop_61_adagrad_batch_normalization_14_beta_accumulator: f
Lassignvariableop_62_adagrad_separable_conv2d_13_depthwise_kernel_accumulator: f
Lassignvariableop_63_adagrad_separable_conv2d_13_pointwise_kernel_accumulator: @N
@assignvariableop_64_adagrad_separable_conv2d_13_bias_accumulator:@R
Dassignvariableop_65_adagrad_batch_normalization_15_gamma_accumulator:@Q
Cassignvariableop_66_adagrad_batch_normalization_15_beta_accumulator:@f
Lassignvariableop_67_adagrad_separable_conv2d_14_depthwise_kernel_accumulator:@f
Lassignvariableop_68_adagrad_separable_conv2d_14_pointwise_kernel_accumulator:@@N
@assignvariableop_69_adagrad_separable_conv2d_14_bias_accumulator:@R
Dassignvariableop_70_adagrad_batch_normalization_16_gamma_accumulator:@Q
Cassignvariableop_71_adagrad_batch_normalization_16_beta_accumulator:@f
Lassignvariableop_72_adagrad_separable_conv2d_15_depthwise_kernel_accumulator:@g
Lassignvariableop_73_adagrad_separable_conv2d_15_pointwise_kernel_accumulator:@O
@assignvariableop_74_adagrad_separable_conv2d_15_bias_accumulator:	S
Dassignvariableop_75_adagrad_batch_normalization_17_gamma_accumulator:	R
Cassignvariableop_76_adagrad_batch_normalization_17_beta_accumulator:	g
Lassignvariableop_77_adagrad_separable_conv2d_16_depthwise_kernel_accumulator:h
Lassignvariableop_78_adagrad_separable_conv2d_16_pointwise_kernel_accumulator:O
@assignvariableop_79_adagrad_separable_conv2d_16_bias_accumulator:	S
Dassignvariableop_80_adagrad_batch_normalization_18_gamma_accumulator:	R
Cassignvariableop_81_adagrad_batch_normalization_18_beta_accumulator:	g
Lassignvariableop_82_adagrad_separable_conv2d_17_depthwise_kernel_accumulator:h
Lassignvariableop_83_adagrad_separable_conv2d_17_pointwise_kernel_accumulator:O
@assignvariableop_84_adagrad_separable_conv2d_17_bias_accumulator:	S
Dassignvariableop_85_adagrad_batch_normalization_19_gamma_accumulator:	R
Cassignvariableop_86_adagrad_batch_normalization_19_beta_accumulator:	J
6assignvariableop_87_adagrad_dense_4_kernel_accumulator:
$C
4assignvariableop_88_adagrad_dense_4_bias_accumulator:	S
Dassignvariableop_89_adagrad_batch_normalization_20_gamma_accumulator:	R
Cassignvariableop_90_adagrad_batch_normalization_20_beta_accumulator:	I
6assignvariableop_91_adagrad_dense_5_kernel_accumulator:	B
4assignvariableop_92_adagrad_dense_5_bias_accumulator:
identity_94¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92Ō6
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*ą5
valueÖ5BÓ5^B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-6/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-6/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBflayer_with_weights-8/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-10/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-10/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesĶ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:^*
dtype0*Ń
valueĒBÄ^B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesū
ų::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*l
dtypesb
`2^	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity“
AssignVariableOpAssignVariableOp5assignvariableop_separable_conv2d_12_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¼
AssignVariableOp_1AssignVariableOp7assignvariableop_1_separable_conv2d_12_pointwise_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOp+assignvariableop_2_separable_conv2d_12_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3“
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_14_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4³
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_14_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ŗ
AssignVariableOp_5AssignVariableOp5assignvariableop_5_batch_normalization_14_moving_meanIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¾
AssignVariableOp_6AssignVariableOp9assignvariableop_6_batch_normalization_14_moving_varianceIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¼
AssignVariableOp_7AssignVariableOp7assignvariableop_7_separable_conv2d_13_depthwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¼
AssignVariableOp_8AssignVariableOp7assignvariableop_8_separable_conv2d_13_pointwise_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9°
AssignVariableOp_9AssignVariableOp+assignvariableop_9_separable_conv2d_13_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ø
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_15_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_15_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¾
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_15_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ā
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_15_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ą
AssignVariableOp_14AssignVariableOp8assignvariableop_14_separable_conv2d_14_depthwise_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ą
AssignVariableOp_15AssignVariableOp8assignvariableop_15_separable_conv2d_14_pointwise_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16“
AssignVariableOp_16AssignVariableOp,assignvariableop_16_separable_conv2d_14_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ø
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_16_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18·
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_16_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¾
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_16_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ā
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_16_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ą
AssignVariableOp_21AssignVariableOp8assignvariableop_21_separable_conv2d_15_depthwise_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ą
AssignVariableOp_22AssignVariableOp8assignvariableop_22_separable_conv2d_15_pointwise_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23“
AssignVariableOp_23AssignVariableOp,assignvariableop_23_separable_conv2d_15_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ø
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_17_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25·
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batch_normalization_17_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_batch_normalization_17_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ā
AssignVariableOp_27AssignVariableOp:assignvariableop_27_batch_normalization_17_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ą
AssignVariableOp_28AssignVariableOp8assignvariableop_28_separable_conv2d_16_depthwise_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ą
AssignVariableOp_29AssignVariableOp8assignvariableop_29_separable_conv2d_16_pointwise_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30“
AssignVariableOp_30AssignVariableOp,assignvariableop_30_separable_conv2d_16_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ø
AssignVariableOp_31AssignVariableOp0assignvariableop_31_batch_normalization_18_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32·
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_18_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¾
AssignVariableOp_33AssignVariableOp6assignvariableop_33_batch_normalization_18_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ā
AssignVariableOp_34AssignVariableOp:assignvariableop_34_batch_normalization_18_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ą
AssignVariableOp_35AssignVariableOp8assignvariableop_35_separable_conv2d_17_depthwise_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ą
AssignVariableOp_36AssignVariableOp8assignvariableop_36_separable_conv2d_17_pointwise_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37“
AssignVariableOp_37AssignVariableOp,assignvariableop_37_separable_conv2d_17_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ø
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_19_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39·
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_19_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¾
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_19_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ā
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_19_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ŗ
AssignVariableOp_42AssignVariableOp"assignvariableop_42_dense_4_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ø
AssignVariableOp_43AssignVariableOp assignvariableop_43_dense_4_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ø
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_20_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45·
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_20_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¾
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_20_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ā
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_20_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ŗ
AssignVariableOp_48AssignVariableOp"assignvariableop_48_dense_5_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ø
AssignVariableOp_49AssignVariableOp assignvariableop_49_dense_5_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_50Ø
AssignVariableOp_50AssignVariableOp assignvariableop_50_adagrad_iterIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51©
AssignVariableOp_51AssignVariableOp!assignvariableop_51_adagrad_decayIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adagrad_learning_rateIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53”
AssignVariableOp_53AssignVariableOpassignvariableop_53_totalIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54”
AssignVariableOp_54AssignVariableOpassignvariableop_54_countIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55£
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56£
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ō
AssignVariableOp_57AssignVariableOpLassignvariableop_57_adagrad_separable_conv2d_12_depthwise_kernel_accumulatorIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ō
AssignVariableOp_58AssignVariableOpLassignvariableop_58_adagrad_separable_conv2d_12_pointwise_kernel_accumulatorIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Č
AssignVariableOp_59AssignVariableOp@assignvariableop_59_adagrad_separable_conv2d_12_bias_accumulatorIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ģ
AssignVariableOp_60AssignVariableOpDassignvariableop_60_adagrad_batch_normalization_14_gamma_accumulatorIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ė
AssignVariableOp_61AssignVariableOpCassignvariableop_61_adagrad_batch_normalization_14_beta_accumulatorIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Ō
AssignVariableOp_62AssignVariableOpLassignvariableop_62_adagrad_separable_conv2d_13_depthwise_kernel_accumulatorIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ō
AssignVariableOp_63AssignVariableOpLassignvariableop_63_adagrad_separable_conv2d_13_pointwise_kernel_accumulatorIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Č
AssignVariableOp_64AssignVariableOp@assignvariableop_64_adagrad_separable_conv2d_13_bias_accumulatorIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ģ
AssignVariableOp_65AssignVariableOpDassignvariableop_65_adagrad_batch_normalization_15_gamma_accumulatorIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Ė
AssignVariableOp_66AssignVariableOpCassignvariableop_66_adagrad_batch_normalization_15_beta_accumulatorIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ō
AssignVariableOp_67AssignVariableOpLassignvariableop_67_adagrad_separable_conv2d_14_depthwise_kernel_accumulatorIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ō
AssignVariableOp_68AssignVariableOpLassignvariableop_68_adagrad_separable_conv2d_14_pointwise_kernel_accumulatorIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Č
AssignVariableOp_69AssignVariableOp@assignvariableop_69_adagrad_separable_conv2d_14_bias_accumulatorIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ģ
AssignVariableOp_70AssignVariableOpDassignvariableop_70_adagrad_batch_normalization_16_gamma_accumulatorIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ė
AssignVariableOp_71AssignVariableOpCassignvariableop_71_adagrad_batch_normalization_16_beta_accumulatorIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ō
AssignVariableOp_72AssignVariableOpLassignvariableop_72_adagrad_separable_conv2d_15_depthwise_kernel_accumulatorIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ō
AssignVariableOp_73AssignVariableOpLassignvariableop_73_adagrad_separable_conv2d_15_pointwise_kernel_accumulatorIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Č
AssignVariableOp_74AssignVariableOp@assignvariableop_74_adagrad_separable_conv2d_15_bias_accumulatorIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Ģ
AssignVariableOp_75AssignVariableOpDassignvariableop_75_adagrad_batch_normalization_17_gamma_accumulatorIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Ė
AssignVariableOp_76AssignVariableOpCassignvariableop_76_adagrad_batch_normalization_17_beta_accumulatorIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Ō
AssignVariableOp_77AssignVariableOpLassignvariableop_77_adagrad_separable_conv2d_16_depthwise_kernel_accumulatorIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Ō
AssignVariableOp_78AssignVariableOpLassignvariableop_78_adagrad_separable_conv2d_16_pointwise_kernel_accumulatorIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Č
AssignVariableOp_79AssignVariableOp@assignvariableop_79_adagrad_separable_conv2d_16_bias_accumulatorIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Ģ
AssignVariableOp_80AssignVariableOpDassignvariableop_80_adagrad_batch_normalization_18_gamma_accumulatorIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81Ė
AssignVariableOp_81AssignVariableOpCassignvariableop_81_adagrad_batch_normalization_18_beta_accumulatorIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82Ō
AssignVariableOp_82AssignVariableOpLassignvariableop_82_adagrad_separable_conv2d_17_depthwise_kernel_accumulatorIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83Ō
AssignVariableOp_83AssignVariableOpLassignvariableop_83_adagrad_separable_conv2d_17_pointwise_kernel_accumulatorIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84Č
AssignVariableOp_84AssignVariableOp@assignvariableop_84_adagrad_separable_conv2d_17_bias_accumulatorIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Ģ
AssignVariableOp_85AssignVariableOpDassignvariableop_85_adagrad_batch_normalization_19_gamma_accumulatorIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86Ė
AssignVariableOp_86AssignVariableOpCassignvariableop_86_adagrad_batch_normalization_19_beta_accumulatorIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87¾
AssignVariableOp_87AssignVariableOp6assignvariableop_87_adagrad_dense_4_kernel_accumulatorIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88¼
AssignVariableOp_88AssignVariableOp4assignvariableop_88_adagrad_dense_4_bias_accumulatorIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89Ģ
AssignVariableOp_89AssignVariableOpDassignvariableop_89_adagrad_batch_normalization_20_gamma_accumulatorIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Ė
AssignVariableOp_90AssignVariableOpCassignvariableop_90_adagrad_batch_normalization_20_beta_accumulatorIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91¾
AssignVariableOp_91AssignVariableOp6assignvariableop_91_adagrad_dense_5_kernel_accumulatorIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92¼
AssignVariableOp_92AssignVariableOp4assignvariableop_92_adagrad_dense_5_bias_accumulatorIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_929
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÜ
Identity_93Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_93Ļ
Identity_94IdentityIdentity_93:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92*
T0*
_output_shapes
: 2
Identity_94"#
identity_94Identity_94:output:0*Ń
_input_shapesæ
¼: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_92:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ķ
c
*__inference_dropout_8_layer_call_fn_323005

inputs
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3213322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Õ
F
*__inference_flatten_2_layer_call_fn_323751

inputs
identityĒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3207992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
Į
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_321368

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ž
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’00 : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
µ
e
F__inference_dropout_11_layer_call_and_return_conditional_losses_320991

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yæ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ķ
c
*__inference_dropout_9_layer_call_fn_323300

inputs
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3212092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ą
Ņ
4__inference_separable_conv2d_16_layer_call_fn_320062

inputs"
unknown:%
	unknown_0:
	unknown_1:	
identity¢StatefulPartitionedCallŖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_3200502
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ü
Š
4__inference_separable_conv2d_15_layer_call_fn_319908

inputs!
unknown:@$
	unknown_0:@
	unknown_1:	
identity¢StatefulPartitionedCallŖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_3198962
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
Ä

R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322977

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’00 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3Ś
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’00 : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs
Š
Å
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323549

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
į
F
*__inference_dropout_9_layer_call_fn_323295

inputs
identityĪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3206602
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
å
e
I__inference_activation_20_layer_call_and_return_conditional_losses_320822

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų
Į
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322995

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ų
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:’’’’’’’’’00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ž
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:’’’’’’’’’00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:’’’’’’’’’00 : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:’’’’’’’’’00 
 
_user_specified_nameinputs

e
I__inference_activation_18_layer_call_and_return_conditional_losses_320715

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

c
E__inference_dropout_9_layer_call_and_return_conditional_losses_320660

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

e
I__inference_activation_18_layer_call_and_return_conditional_losses_323461

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
é
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_323757

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Š
Å
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_320282

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ļ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

·
H__inference_sequential_2_layer_call_and_return_conditional_losses_321968
separable_conv2d_12_input4
separable_conv2d_12_321837:4
separable_conv2d_12_321839: (
separable_conv2d_12_321841: +
batch_normalization_14_321845: +
batch_normalization_14_321847: +
batch_normalization_14_321849: +
batch_normalization_14_321851: 4
separable_conv2d_13_321856: 4
separable_conv2d_13_321858: @(
separable_conv2d_13_321860:@+
batch_normalization_15_321864:@+
batch_normalization_15_321866:@+
batch_normalization_15_321868:@+
batch_normalization_15_321870:@4
separable_conv2d_14_321873:@4
separable_conv2d_14_321875:@@(
separable_conv2d_14_321877:@+
batch_normalization_16_321881:@+
batch_normalization_16_321883:@+
batch_normalization_16_321885:@+
batch_normalization_16_321887:@4
separable_conv2d_15_321892:@5
separable_conv2d_15_321894:@)
separable_conv2d_15_321896:	,
batch_normalization_17_321900:	,
batch_normalization_17_321902:	,
batch_normalization_17_321904:	,
batch_normalization_17_321906:	5
separable_conv2d_16_321909:6
separable_conv2d_16_321911:)
separable_conv2d_16_321913:	,
batch_normalization_18_321917:	,
batch_normalization_18_321919:	,
batch_normalization_18_321921:	,
batch_normalization_18_321923:	5
separable_conv2d_17_321926:6
separable_conv2d_17_321928:)
separable_conv2d_17_321930:	,
batch_normalization_19_321934:	,
batch_normalization_19_321936:	,
batch_normalization_19_321938:	,
batch_normalization_19_321940:	"
dense_4_321946:
$
dense_4_321948:	,
batch_normalization_20_321952:	,
batch_normalization_20_321954:	,
batch_normalization_20_321956:	,
batch_normalization_20_321958:	!
dense_5_321962:	
dense_5_321964:
identity¢.batch_normalization_14/StatefulPartitionedCall¢.batch_normalization_15/StatefulPartitionedCall¢.batch_normalization_16/StatefulPartitionedCall¢.batch_normalization_17/StatefulPartitionedCall¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢+separable_conv2d_12/StatefulPartitionedCall¢+separable_conv2d_13/StatefulPartitionedCall¢+separable_conv2d_14/StatefulPartitionedCall¢+separable_conv2d_15/StatefulPartitionedCall¢+separable_conv2d_16/StatefulPartitionedCall¢+separable_conv2d_17/StatefulPartitionedCall
+separable_conv2d_12/StatefulPartitionedCallStatefulPartitionedCallseparable_conv2d_12_inputseparable_conv2d_12_321837separable_conv2d_12_321839separable_conv2d_12_321841*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_3194102-
+separable_conv2d_12/StatefulPartitionedCall
activation_14/PartitionedCallPartitionedCall4separable_conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_14_layer_call_and_return_conditional_losses_3205352
activation_14/PartitionedCallĒ
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall&activation_14/PartitionedCall:output:0batch_normalization_14_321845batch_normalization_14_321847batch_normalization_14_321849batch_normalization_14_321851*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_32055420
.batch_normalization_14/StatefulPartitionedCall„
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_3195542!
max_pooling2d_6/PartitionedCall
dropout_8/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_3205702
dropout_8/PartitionedCall
+separable_conv2d_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0separable_conv2d_13_321856separable_conv2d_13_321858separable_conv2d_13_321860*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_3195762-
+separable_conv2d_13/StatefulPartitionedCall
activation_15/PartitionedCallPartitionedCall4separable_conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_15_layer_call_and_return_conditional_losses_3205842
activation_15/PartitionedCallĒ
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall&activation_15/PartitionedCall:output:0batch_normalization_15_321864batch_normalization_15_321866batch_normalization_15_321868batch_normalization_15_321870*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_32060320
.batch_normalization_15/StatefulPartitionedCall„
+separable_conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0separable_conv2d_14_321873separable_conv2d_14_321875separable_conv2d_14_321877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_3197302-
+separable_conv2d_14/StatefulPartitionedCall
activation_16/PartitionedCallPartitionedCall4separable_conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_16_layer_call_and_return_conditional_losses_3206252
activation_16/PartitionedCallĒ
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall&activation_16/PartitionedCall:output:0batch_normalization_16_321881batch_normalization_16_321883batch_normalization_16_321885batch_normalization_16_321887*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_32064420
.batch_normalization_16/StatefulPartitionedCall„
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_3198742!
max_pooling2d_7/PartitionedCall
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3206602
dropout_9/PartitionedCall
+separable_conv2d_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0separable_conv2d_15_321892separable_conv2d_15_321894separable_conv2d_15_321896*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_3198962-
+separable_conv2d_15/StatefulPartitionedCall
activation_17/PartitionedCallPartitionedCall4separable_conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_17_layer_call_and_return_conditional_losses_3206742
activation_17/PartitionedCallČ
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall&activation_17/PartitionedCall:output:0batch_normalization_17_321900batch_normalization_17_321902batch_normalization_17_321904batch_normalization_17_321906*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_32069320
.batch_normalization_17/StatefulPartitionedCall¦
+separable_conv2d_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0separable_conv2d_16_321909separable_conv2d_16_321911separable_conv2d_16_321913*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_3200502-
+separable_conv2d_16/StatefulPartitionedCall
activation_18/PartitionedCallPartitionedCall4separable_conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_3207152
activation_18/PartitionedCallČ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0batch_normalization_18_321917batch_normalization_18_321919batch_normalization_18_321921batch_normalization_18_321923*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_32073420
.batch_normalization_18/StatefulPartitionedCall¦
+separable_conv2d_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0separable_conv2d_17_321926separable_conv2d_17_321928separable_conv2d_17_321930*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_3202042-
+separable_conv2d_17/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall4separable_conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_3207562
activation_19/PartitionedCallČ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0batch_normalization_19_321934batch_normalization_19_321936batch_normalization_19_321938batch_normalization_19_321940*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_32077520
.batch_normalization_19/StatefulPartitionedCall¦
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3203482!
max_pooling2d_8/PartitionedCall
dropout_10/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_3207912
dropout_10/PartitionedCallų
flatten_2/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_3207992
flatten_2/PartitionedCallÆ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_4_321946dense_4_321948*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_3208112!
dense_4/StatefulPartitionedCall
activation_20/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_3208222
activation_20/PartitionedCallĄ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0batch_normalization_20_321952batch_normalization_20_321954batch_normalization_20_321956batch_normalization_20_321958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_32037820
.batch_normalization_20/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_3208382
dropout_11/PartitionedCallÆ
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_321962dense_5_321964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3208512!
dense_5/StatefulPartitionedCall«
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall,^separable_conv2d_12/StatefulPartitionedCall,^separable_conv2d_13/StatefulPartitionedCall,^separable_conv2d_14/StatefulPartitionedCall,^separable_conv2d_15/StatefulPartitionedCall,^separable_conv2d_16/StatefulPartitionedCall,^separable_conv2d_17/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:’’’’’’’’’00: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2Z
+separable_conv2d_12/StatefulPartitionedCall+separable_conv2d_12/StatefulPartitionedCall2Z
+separable_conv2d_13/StatefulPartitionedCall+separable_conv2d_13/StatefulPartitionedCall2Z
+separable_conv2d_14/StatefulPartitionedCall+separable_conv2d_14/StatefulPartitionedCall2Z
+separable_conv2d_15/StatefulPartitionedCall+separable_conv2d_15/StatefulPartitionedCall2Z
+separable_conv2d_16/StatefulPartitionedCall+separable_conv2d_16/StatefulPartitionedCall2Z
+separable_conv2d_17/StatefulPartitionedCall+separable_conv2d_17/StatefulPartitionedCall:j f
/
_output_shapes
:’’’’’’’’’00
3
_user_specified_nameseparable_conv2d_12_input
Ą
Į
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_319808

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
ķ
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_323317

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ŖŖ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yĘ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’@:W S
/
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

e
I__inference_activation_19_layer_call_and_return_conditional_losses_323595

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323236

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3ģ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
ö
Ö
7__inference_batch_normalization_17_layer_call_fn_323340

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_3199302
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ś
L
0__inference_max_pooling2d_8_layer_call_fn_320354

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_3203482
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

e
I__inference_activation_17_layer_call_and_return_conditional_losses_323327

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:’’’’’’’’’2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:’’’’’’’’’:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
®
Ö
7__inference_batch_normalization_18_layer_call_fn_323500

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_3207342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Å
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_321172

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1Ø
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ż
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1’
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
«
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_319554

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ą
Į
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322959

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ź
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Ā
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueĪ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 
_user_specified_nameinputs"ĢL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ö
serving_defaultĀ
g
separable_conv2d_12_inputJ
+serving_default_separable_conv2d_12_input:0’’’’’’’’’00;
dense_50
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:Ü
Ś÷
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer-26
layer_with_weights-13
layer-27
layer-28
layer_with_weights-14
layer-29
	optimizer
 	variables
!regularization_losses
"trainable_variables
#	keras_api
$
signatures
£_default_save_signature
¤__call__
+„&call_and_return_all_conditional_losses"ęļ
_tf_keras_sequentialĘļ{"name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "separable_conv2d_12_input"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 87, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}, "shared_object_id": 88}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 48, 48, 3]}, "float32", "separable_conv2d_12_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "separable_conv2d_12_input"}, "shared_object_id": 0}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 5}, {"class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 6}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 11}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 12}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 13}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 18}, {"class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 19}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 21}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 24}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 29}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 30}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 32}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 34}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 35}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 36}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 37}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 42}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 43}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 45}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 47}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 48}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 53}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 54}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 55}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 56}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 58}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 59}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 63}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 60}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 64}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 65}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 66}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 67}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 69}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 70}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 71}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 72}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 73}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 74}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 75}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 76}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 77}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 78}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 79}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 80}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 81}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 82}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 83}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 84}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 85}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 86}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 89}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.009999999776482582, "decay": 0.00025, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
ų
%depthwise_kernel
&pointwise_kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"±
_tf_keras_layer{"name": "separable_conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}, "shared_object_id": 88}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 3]}}
š
,	variables
-regularization_losses
.trainable_variables
/	keras_api
Ø__call__
+©&call_and_return_all_conditional_losses"ß
_tf_keras_layerÅ{"name": "activation_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_14", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 6}
Ė

0axis
	1gamma
2beta
3moving_mean
4moving_variance
5	variables
6regularization_losses
7trainable_variables
8	keras_api
Ŗ__call__
+«&call_and_return_all_conditional_losses"õ
_tf_keras_layerŪ{"name": "batch_normalization_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_14", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 90}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 32]}}
±
9	variables
:regularization_losses
;trainable_variables
<	keras_api
¬__call__
+­&call_and_return_all_conditional_losses" 
_tf_keras_layer{"name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 91}}

=	variables
>regularization_losses
?trainable_variables
@	keras_api
®__call__
+Æ&call_and_return_all_conditional_losses"ļ
_tf_keras_layerÕ{"name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 13}
ž
Adepthwise_kernel
Bpointwise_kernel
Cbias
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
°__call__
+±&call_and_return_all_conditional_losses"·
_tf_keras_layer{"name": "separable_conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 92}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}}
ń
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
²__call__
+³&call_and_return_all_conditional_losses"ą
_tf_keras_layerĘ{"name": "activation_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_15", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 19}
Ī

Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
“__call__
+µ&call_and_return_all_conditional_losses"ų
_tf_keras_layerŽ{"name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 21}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 93}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 64]}}
ž
Udepthwise_kernel
Vpointwise_kernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"·
_tf_keras_layer{"name": "separable_conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 94}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 64]}}
ń
\	variables
]regularization_losses
^trainable_variables
_	keras_api
ø__call__
+¹&call_and_return_all_conditional_losses"ą
_tf_keras_layerĘ{"name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 30}
Ī

`axis
	agamma
bbeta
cmoving_mean
dmoving_variance
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
ŗ__call__
+»&call_and_return_all_conditional_losses"ų
_tf_keras_layerŽ{"name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 32}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 34}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 95}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 64]}}
±
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
¼__call__
+½&call_and_return_all_conditional_losses" 
_tf_keras_layer{"name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 96}}

m	variables
nregularization_losses
otrainable_variables
p	keras_api
¾__call__
+æ&call_and_return_all_conditional_losses"ļ
_tf_keras_layerÕ{"name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 37}
’
qdepthwise_kernel
rpointwise_kernel
sbias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
Ą__call__
+Į&call_and_return_all_conditional_losses"ø
_tf_keras_layer{"name": "separable_conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 97}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 64]}}
ń
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
Ā__call__
+Ć&call_and_return_all_conditional_losses"ą
_tf_keras_layerĘ{"name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 43}
Õ

|axis
	}gamma
~beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"ś
_tf_keras_layerą{"name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 45}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 47}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 98}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}

depthwise_kernel
pointwise_kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
Ę__call__
+Ē&call_and_return_all_conditional_losses"ŗ
_tf_keras_layer {"name": "separable_conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 53, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}, "shared_object_id": 99}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}
õ
	variables
regularization_losses
trainable_variables
	keras_api
Č__call__
+É&call_and_return_all_conditional_losses"ą
_tf_keras_layerĘ{"name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 54}
Ś

	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
Ź__call__
+Ė&call_and_return_all_conditional_losses"ū
_tf_keras_layerį{"name": "batch_normalization_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 55}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 56}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 58}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 100}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}

depthwise_kernel
pointwise_kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
Ģ__call__
+Ķ&call_and_return_all_conditional_losses"»
_tf_keras_layer”{"name": "separable_conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 63}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 60}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "shared_object_id": 64, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}, "shared_object_id": 101}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}
õ
 	variables
”regularization_losses
¢trainable_variables
£	keras_api
Ī__call__
+Ļ&call_and_return_all_conditional_losses"ą
_tf_keras_layerĘ{"name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 65}
Ś

	¤axis

„gamma
	¦beta
§moving_mean
Ømoving_variance
©	variables
Ŗregularization_losses
«trainable_variables
¬	keras_api
Š__call__
+Ń&call_and_return_all_conditional_losses"ū
_tf_keras_layerį{"name": "batch_normalization_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 66}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 67}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 69}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 70, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 102}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}
¶
­	variables
®regularization_losses
Ætrainable_variables
°	keras_api
Ņ__call__
+Ó&call_and_return_all_conditional_losses"”
_tf_keras_layer{"name": "max_pooling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 71, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 103}}

±	variables
²regularization_losses
³trainable_variables
“	keras_api
Ō__call__
+Õ&call_and_return_all_conditional_losses"ń
_tf_keras_layer×{"name": "dropout_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 72}

µ	variables
¶regularization_losses
·trainable_variables
ø	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"
_tf_keras_layerī{"name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 73, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 104}}
ą
¹kernel
	ŗbias
»	variables
¼regularization_losses
½trainable_variables
¾	keras_api
Ų__call__
+Ł&call_and_return_all_conditional_losses"³
_tf_keras_layer{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 74}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 75}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 76, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4608}}, "shared_object_id": 105}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4608]}}
õ
æ	variables
Ąregularization_losses
Įtrainable_variables
Ā	keras_api
Ś__call__
+Ū&call_and_return_all_conditional_losses"ą
_tf_keras_layerĘ{"name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 77}
Ņ

	Ćaxis

Ägamma
	Åbeta
Ęmoving_mean
Ēmoving_variance
Č	variables
Éregularization_losses
Źtrainable_variables
Ė	keras_api
Ü__call__
+Ż&call_and_return_all_conditional_losses"ó
_tf_keras_layerŁ{"name": "batch_normalization_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 78}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 79}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 80}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 81}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 82, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}, "shared_object_id": 106}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}

Ģ	variables
Ķregularization_losses
Ītrainable_variables
Ļ	keras_api
Ž__call__
+ß&call_and_return_all_conditional_losses"š
_tf_keras_layerÖ{"name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 83}
Ż
Škernel
	Ńbias
Ņ	variables
Óregularization_losses
Ōtrainable_variables
Õ	keras_api
ą__call__
+į&call_and_return_all_conditional_losses"°
_tf_keras_layer{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 84}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 85}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 86, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 107}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}

	Öiter

×decay
Ųlearning_rate%accumulator’&accumulator'accumulator1accumulator2accumulatorAaccumulatorBaccumulatorCaccumulatorMaccumulatorNaccumulatorUaccumulatorVaccumulatorWaccumulatoraaccumulatorbaccumulatorqaccumulatorraccumulatorsaccumulator}accumulator~accumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulatoraccumulator„accumulator¦accumulator¹accumulatorŗaccumulatorÄaccumulatorÅaccumulator Šaccumulator”Ńaccumulator¢"
	optimizer
½
%0
&1
'2
13
24
35
46
A7
B8
C9
M10
N11
O12
P13
U14
V15
W16
a17
b18
c19
d20
q21
r22
s23
}24
~25
26
27
28
29
30
31
32
33
34
35
36
37
„38
¦39
§40
Ø41
¹42
ŗ43
Ä44
Å45
Ę46
Ē47
Š48
Ń49"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
%0
&1
'2
13
24
A5
B6
C7
M8
N9
U10
V11
W12
a13
b14
q15
r16
s17
}18
~19
20
21
22
23
24
25
26
27
„28
¦29
¹30
ŗ31
Ä32
Å33
Š34
Ń35"
trackable_list_wrapper
Ó
Łnon_trainable_variables
Ślayer_metrics
 	variables
 Ūlayer_regularization_losses
!regularization_losses
Ümetrics
Żlayers
"trainable_variables
¤__call__
£_default_save_signature
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
-
āserving_default"
signature_map
>:<2$separable_conv2d_12/depthwise_kernel
>:< 2$separable_conv2d_12/pointwise_kernel
&:$ 2separable_conv2d_12/bias
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
µ
Žnon_trainable_variables
ßlayer_metrics
(	variables
 ąlayer_regularization_losses
)regularization_losses
įmetrics
ālayers
*trainable_variables
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ćnon_trainable_variables
älayer_metrics
,	variables
 ålayer_regularization_losses
-regularization_losses
ęmetrics
ēlayers
.trainable_variables
Ø__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_14/gamma
):' 2batch_normalization_14/beta
2:0  (2"batch_normalization_14/moving_mean
6:4  (2&batch_normalization_14/moving_variance
<
10
21
32
43"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
µ
čnon_trainable_variables
élayer_metrics
5	variables
 źlayer_regularization_losses
6regularization_losses
ėmetrics
ģlayers
7trainable_variables
Ŗ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ķnon_trainable_variables
īlayer_metrics
9	variables
 ļlayer_regularization_losses
:regularization_losses
šmetrics
ńlayers
;trainable_variables
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ņnon_trainable_variables
ólayer_metrics
=	variables
 ōlayer_regularization_losses
>regularization_losses
õmetrics
ölayers
?trainable_variables
®__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
>:< 2$separable_conv2d_13/depthwise_kernel
>:< @2$separable_conv2d_13/pointwise_kernel
&:$@2separable_conv2d_13/bias
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
µ
÷non_trainable_variables
ųlayer_metrics
D	variables
 łlayer_regularization_losses
Eregularization_losses
śmetrics
ūlayers
Ftrainable_variables
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ünon_trainable_variables
żlayer_metrics
H	variables
 žlayer_regularization_losses
Iregularization_losses
’metrics
layers
Jtrainable_variables
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_15/gamma
):'@2batch_normalization_15/beta
2:0@ (2"batch_normalization_15/moving_mean
6:4@ (2&batch_normalization_15/moving_variance
<
M0
N1
O2
P3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Q	variables
 layer_regularization_losses
Rregularization_losses
metrics
layers
Strainable_variables
“__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
>:<@2$separable_conv2d_14/depthwise_kernel
>:<@@2$separable_conv2d_14/pointwise_kernel
&:$@2separable_conv2d_14/bias
5
U0
V1
W2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
U0
V1
W2"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
X	variables
 layer_regularization_losses
Yregularization_losses
metrics
layers
Ztrainable_variables
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
\	variables
 layer_regularization_losses
]regularization_losses
metrics
layers
^trainable_variables
ø__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_16/gamma
):'@2batch_normalization_16/beta
2:0@ (2"batch_normalization_16/moving_mean
6:4@ (2&batch_normalization_16/moving_variance
<
a0
b1
c2
d3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
e	variables
 layer_regularization_losses
fregularization_losses
metrics
layers
gtrainable_variables
ŗ__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
i	variables
 layer_regularization_losses
jregularization_losses
metrics
layers
ktrainable_variables
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
m	variables
 layer_regularization_losses
nregularization_losses
metrics
layers
otrainable_variables
¾__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
>:<@2$separable_conv2d_15/depthwise_kernel
?:=@2$separable_conv2d_15/pointwise_kernel
':%2separable_conv2d_15/bias
5
q0
r1
s2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
q0
r1
s2"
trackable_list_wrapper
µ
non_trainable_variables
 layer_metrics
t	variables
 ”layer_regularization_losses
uregularization_losses
¢metrics
£layers
vtrainable_variables
Ą__call__
+Į&call_and_return_all_conditional_losses
'Į"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¤non_trainable_variables
„layer_metrics
x	variables
 ¦layer_regularization_losses
yregularization_losses
§metrics
Ølayers
ztrainable_variables
Ā__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_17/gamma
*:(2batch_normalization_17/beta
3:1 (2"batch_normalization_17/moving_mean
7:5 (2&batch_normalization_17/moving_variance
=
}0
~1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
ø
©non_trainable_variables
Ŗlayer_metrics
	variables
 «layer_regularization_losses
regularization_losses
¬metrics
­layers
trainable_variables
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
?:=2$separable_conv2d_16/depthwise_kernel
@:>2$separable_conv2d_16/pointwise_kernel
':%2separable_conv2d_16/bias
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
ø
®non_trainable_variables
Ælayer_metrics
	variables
 °layer_regularization_losses
regularization_losses
±metrics
²layers
trainable_variables
Ę__call__
+Ē&call_and_return_all_conditional_losses
'Ē"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
³non_trainable_variables
“layer_metrics
	variables
 µlayer_regularization_losses
regularization_losses
¶metrics
·layers
trainable_variables
Č__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_18/gamma
*:(2batch_normalization_18/beta
3:1 (2"batch_normalization_18/moving_mean
7:5 (2&batch_normalization_18/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
ø
ønon_trainable_variables
¹layer_metrics
	variables
 ŗlayer_regularization_losses
regularization_losses
»metrics
¼layers
trainable_variables
Ź__call__
+Ė&call_and_return_all_conditional_losses
'Ė"call_and_return_conditional_losses"
_generic_user_object
?:=2$separable_conv2d_17/depthwise_kernel
@:>2$separable_conv2d_17/pointwise_kernel
':%2separable_conv2d_17/bias
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
ø
½non_trainable_variables
¾layer_metrics
	variables
 ælayer_regularization_losses
regularization_losses
Ąmetrics
Įlayers
trainable_variables
Ģ__call__
+Ķ&call_and_return_all_conditional_losses
'Ķ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ānon_trainable_variables
Ćlayer_metrics
 	variables
 Älayer_regularization_losses
”regularization_losses
Åmetrics
Ęlayers
¢trainable_variables
Ī__call__
+Ļ&call_and_return_all_conditional_losses
'Ļ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_19/gamma
*:(2batch_normalization_19/beta
3:1 (2"batch_normalization_19/moving_mean
7:5 (2&batch_normalization_19/moving_variance
@
„0
¦1
§2
Ø3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
„0
¦1"
trackable_list_wrapper
ø
Ēnon_trainable_variables
Člayer_metrics
©	variables
 Élayer_regularization_losses
Ŗregularization_losses
Źmetrics
Ėlayers
«trainable_variables
Š__call__
+Ń&call_and_return_all_conditional_losses
'Ń"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ģnon_trainable_variables
Ķlayer_metrics
­	variables
 Īlayer_regularization_losses
®regularization_losses
Ļmetrics
Šlayers
Ætrainable_variables
Ņ__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ńnon_trainable_variables
Ņlayer_metrics
±	variables
 Ólayer_regularization_losses
²regularization_losses
Ōmetrics
Õlayers
³trainable_variables
Ō__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Önon_trainable_variables
×layer_metrics
µ	variables
 Ųlayer_regularization_losses
¶regularization_losses
Łmetrics
Ślayers
·trainable_variables
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
": 
$2dense_4/kernel
:2dense_4/bias
0
¹0
ŗ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¹0
ŗ1"
trackable_list_wrapper
ø
Ūnon_trainable_variables
Ülayer_metrics
»	variables
 Żlayer_regularization_losses
¼regularization_losses
Žmetrics
ßlayers
½trainable_variables
Ų__call__
+Ł&call_and_return_all_conditional_losses
'Ł"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ąnon_trainable_variables
įlayer_metrics
æ	variables
 ālayer_regularization_losses
Ąregularization_losses
ćmetrics
älayers
Įtrainable_variables
Ś__call__
+Ū&call_and_return_all_conditional_losses
'Ū"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_20/gamma
*:(2batch_normalization_20/beta
3:1 (2"batch_normalization_20/moving_mean
7:5 (2&batch_normalization_20/moving_variance
@
Ä0
Å1
Ę2
Ē3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ä0
Å1"
trackable_list_wrapper
ø
ånon_trainable_variables
ęlayer_metrics
Č	variables
 ēlayer_regularization_losses
Éregularization_losses
čmetrics
élayers
Źtrainable_variables
Ü__call__
+Ż&call_and_return_all_conditional_losses
'Ż"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
źnon_trainable_variables
ėlayer_metrics
Ģ	variables
 ģlayer_regularization_losses
Ķregularization_losses
ķmetrics
īlayers
Ītrainable_variables
Ž__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_5/kernel
:2dense_5/bias
0
Š0
Ń1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Š0
Ń1"
trackable_list_wrapper
ø
ļnon_trainable_variables
šlayer_metrics
Ņ	variables
 ńlayer_regularization_losses
Óregularization_losses
ņmetrics
ólayers
Ōtrainable_variables
ą__call__
+į&call_and_return_all_conditional_losses
'į"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate

30
41
O2
P3
c4
d5
6
7
8
9
§10
Ø11
Ę12
Ē13"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
ō0
õ1"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
§0
Ø1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ę0
Ē1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ł

ötotal

÷count
ų	variables
ł	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 108}


śtotal

ūcount
ü
_fn_kwargs
ż	variables
ž	keras_api"Ė
_tf_keras_metric°{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 89}
:  (2total
:  (2count
0
ö0
÷1"
trackable_list_wrapper
.
ų	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ś0
ū1"
trackable_list_wrapper
.
ż	variables"
_generic_user_object
P:N28Adagrad/separable_conv2d_12/depthwise_kernel/accumulator
P:N 28Adagrad/separable_conv2d_12/pointwise_kernel/accumulator
8:6 2,Adagrad/separable_conv2d_12/bias/accumulator
<:: 20Adagrad/batch_normalization_14/gamma/accumulator
;:9 2/Adagrad/batch_normalization_14/beta/accumulator
P:N 28Adagrad/separable_conv2d_13/depthwise_kernel/accumulator
P:N @28Adagrad/separable_conv2d_13/pointwise_kernel/accumulator
8:6@2,Adagrad/separable_conv2d_13/bias/accumulator
<::@20Adagrad/batch_normalization_15/gamma/accumulator
;:9@2/Adagrad/batch_normalization_15/beta/accumulator
P:N@28Adagrad/separable_conv2d_14/depthwise_kernel/accumulator
P:N@@28Adagrad/separable_conv2d_14/pointwise_kernel/accumulator
8:6@2,Adagrad/separable_conv2d_14/bias/accumulator
<::@20Adagrad/batch_normalization_16/gamma/accumulator
;:9@2/Adagrad/batch_normalization_16/beta/accumulator
P:N@28Adagrad/separable_conv2d_15/depthwise_kernel/accumulator
Q:O@28Adagrad/separable_conv2d_15/pointwise_kernel/accumulator
9:72,Adagrad/separable_conv2d_15/bias/accumulator
=:;20Adagrad/batch_normalization_17/gamma/accumulator
<::2/Adagrad/batch_normalization_17/beta/accumulator
Q:O28Adagrad/separable_conv2d_16/depthwise_kernel/accumulator
R:P28Adagrad/separable_conv2d_16/pointwise_kernel/accumulator
9:72,Adagrad/separable_conv2d_16/bias/accumulator
=:;20Adagrad/batch_normalization_18/gamma/accumulator
<::2/Adagrad/batch_normalization_18/beta/accumulator
Q:O28Adagrad/separable_conv2d_17/depthwise_kernel/accumulator
R:P28Adagrad/separable_conv2d_17/pointwise_kernel/accumulator
9:72,Adagrad/separable_conv2d_17/bias/accumulator
=:;20Adagrad/batch_normalization_19/gamma/accumulator
<::2/Adagrad/batch_normalization_19/beta/accumulator
4:2
$2"Adagrad/dense_4/kernel/accumulator
-:+2 Adagrad/dense_4/bias/accumulator
=:;20Adagrad/batch_normalization_20/gamma/accumulator
<::2/Adagrad/batch_normalization_20/beta/accumulator
3:1	2"Adagrad/dense_5/kernel/accumulator
,:*2 Adagrad/dense_5/bias/accumulator
ł2ö
!__inference__wrapped_model_319394Š
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;8
separable_conv2d_12_input’’’’’’’’’00
2’
-__inference_sequential_2_layer_call_fn_320961
-__inference_sequential_2_layer_call_fn_322316
-__inference_sequential_2_layer_call_fn_322421
-__inference_sequential_2_layer_call_fn_321834Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ī2ė
H__inference_sequential_2_layer_call_and_return_conditional_losses_322620
H__inference_sequential_2_layer_call_and_return_conditional_losses_322861
H__inference_sequential_2_layer_call_and_return_conditional_losses_321968
H__inference_sequential_2_layer_call_and_return_conditional_losses_322102Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
4__inference_separable_conv2d_12_layer_call_fn_319422×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’
®2«
O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_319410×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ų2Õ
.__inference_activation_14_layer_call_fn_322866¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_activation_14_layer_call_and_return_conditional_losses_322871¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_14_layer_call_fn_322884
7__inference_batch_normalization_14_layer_call_fn_322897
7__inference_batch_normalization_14_layer_call_fn_322910
7__inference_batch_normalization_14_layer_call_fn_322923“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322941
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322959
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322977
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322995“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
0__inference_max_pooling2d_6_layer_call_fn_319560ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
³2°
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_319554ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
*__inference_dropout_8_layer_call_fn_323000
*__inference_dropout_8_layer_call_fn_323005“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Č2Å
E__inference_dropout_8_layer_call_and_return_conditional_losses_323010
E__inference_dropout_8_layer_call_and_return_conditional_losses_323022“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
4__inference_separable_conv2d_13_layer_call_fn_319588×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
®2«
O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_319576×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
Ų2Õ
.__inference_activation_15_layer_call_fn_323027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_activation_15_layer_call_and_return_conditional_losses_323032¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_15_layer_call_fn_323045
7__inference_batch_normalization_15_layer_call_fn_323058
7__inference_batch_normalization_15_layer_call_fn_323071
7__inference_batch_normalization_15_layer_call_fn_323084“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323102
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323120
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323138
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323156“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
4__inference_separable_conv2d_14_layer_call_fn_319742×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
®2«
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_319730×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
Ų2Õ
.__inference_activation_16_layer_call_fn_323161¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_activation_16_layer_call_and_return_conditional_losses_323166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_16_layer_call_fn_323179
7__inference_batch_normalization_16_layer_call_fn_323192
7__inference_batch_normalization_16_layer_call_fn_323205
7__inference_batch_normalization_16_layer_call_fn_323218“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323236
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323254
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323272
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323290“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
0__inference_max_pooling2d_7_layer_call_fn_319880ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
³2°
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_319874ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
*__inference_dropout_9_layer_call_fn_323295
*__inference_dropout_9_layer_call_fn_323300“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Č2Å
E__inference_dropout_9_layer_call_and_return_conditional_losses_323305
E__inference_dropout_9_layer_call_and_return_conditional_losses_323317“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
4__inference_separable_conv2d_15_layer_call_fn_319908×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
®2«
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_319896×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
Ų2Õ
.__inference_activation_17_layer_call_fn_323322¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_activation_17_layer_call_and_return_conditional_losses_323327¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_17_layer_call_fn_323340
7__inference_batch_normalization_17_layer_call_fn_323353
7__inference_batch_normalization_17_layer_call_fn_323366
7__inference_batch_normalization_17_layer_call_fn_323379“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323397
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323415
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323433
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323451“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
4__inference_separable_conv2d_16_layer_call_fn_320062Ų
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *8¢5
30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Æ2¬
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_320050Ų
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *8¢5
30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ų2Õ
.__inference_activation_18_layer_call_fn_323456¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_activation_18_layer_call_and_return_conditional_losses_323461¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_18_layer_call_fn_323474
7__inference_batch_normalization_18_layer_call_fn_323487
7__inference_batch_normalization_18_layer_call_fn_323500
7__inference_batch_normalization_18_layer_call_fn_323513“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323531
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323549
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323567
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323585“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
4__inference_separable_conv2d_17_layer_call_fn_320216Ų
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *8¢5
30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Æ2¬
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_320204Ų
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *8¢5
30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ų2Õ
.__inference_activation_19_layer_call_fn_323590¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_activation_19_layer_call_and_return_conditional_losses_323595¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_19_layer_call_fn_323608
7__inference_batch_normalization_19_layer_call_fn_323621
7__inference_batch_normalization_19_layer_call_fn_323634
7__inference_batch_normalization_19_layer_call_fn_323647“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323665
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323683
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323701
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323719“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
0__inference_max_pooling2d_8_layer_call_fn_320354ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
³2°
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_320348ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
+__inference_dropout_10_layer_call_fn_323724
+__inference_dropout_10_layer_call_fn_323729“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ź2Ē
F__inference_dropout_10_layer_call_and_return_conditional_losses_323734
F__inference_dropout_10_layer_call_and_return_conditional_losses_323746“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ō2Ń
*__inference_flatten_2_layer_call_fn_323751¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_flatten_2_layer_call_and_return_conditional_losses_323757¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_4_layer_call_fn_323766¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_dense_4_layer_call_and_return_conditional_losses_323776¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ų2Õ
.__inference_activation_20_layer_call_fn_323781¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_activation_20_layer_call_and_return_conditional_losses_323786¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
¬2©
7__inference_batch_normalization_20_layer_call_fn_323799
7__inference_batch_normalization_20_layer_call_fn_323812“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ß
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_323832
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_323866“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
+__inference_dropout_11_layer_call_fn_323871
+__inference_dropout_11_layer_call_fn_323876“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ź2Ē
F__inference_dropout_11_layer_call_and_return_conditional_losses_323881
F__inference_dropout_11_layer_call_and_return_conditional_losses_323893“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
(__inference_dense_5_layer_call_fn_323902¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_dense_5_layer_call_and_return_conditional_losses_323913¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ŻBŚ
$__inference_signature_wrapper_322211separable_conv2d_12_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 š
!__inference__wrapped_model_319394ŹI%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĒÄĘÅŠŃJ¢G
@¢=
;8
separable_conv2d_12_input’’’’’’’’’00
Ŗ "1Ŗ.
,
dense_5!
dense_5’’’’’’’’’µ
I__inference_activation_14_layer_call_and_return_conditional_losses_322871h7¢4
-¢*
(%
inputs’’’’’’’’’00 
Ŗ "-¢*
# 
0’’’’’’’’’00 
 
.__inference_activation_14_layer_call_fn_322866[7¢4
-¢*
(%
inputs’’’’’’’’’00 
Ŗ " ’’’’’’’’’00 µ
I__inference_activation_15_layer_call_and_return_conditional_losses_323032h7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ "-¢*
# 
0’’’’’’’’’@
 
.__inference_activation_15_layer_call_fn_323027[7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ " ’’’’’’’’’@µ
I__inference_activation_16_layer_call_and_return_conditional_losses_323166h7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ "-¢*
# 
0’’’’’’’’’@
 
.__inference_activation_16_layer_call_fn_323161[7¢4
-¢*
(%
inputs’’’’’’’’’@
Ŗ " ’’’’’’’’’@·
I__inference_activation_17_layer_call_and_return_conditional_losses_323327j8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
.__inference_activation_17_layer_call_fn_323322]8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’·
I__inference_activation_18_layer_call_and_return_conditional_losses_323461j8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
.__inference_activation_18_layer_call_fn_323456]8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’·
I__inference_activation_19_layer_call_and_return_conditional_losses_323595j8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ ".¢+
$!
0’’’’’’’’’
 
.__inference_activation_19_layer_call_fn_323590]8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "!’’’’’’’’’§
I__inference_activation_20_layer_call_and_return_conditional_losses_323786Z0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
.__inference_activation_20_layer_call_fn_323781M0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ķ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3229411234M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
p 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 ķ
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3229591234M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
p
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 Č
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322977r1234;¢8
1¢.
(%
inputs’’’’’’’’’00 
p 
Ŗ "-¢*
# 
0’’’’’’’’’00 
 Č
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_322995r1234;¢8
1¢.
(%
inputs’’’’’’’’’00 
p
Ŗ "-¢*
# 
0’’’’’’’’’00 
 Å
7__inference_batch_normalization_14_layer_call_fn_3228841234M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
p 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’ Å
7__inference_batch_normalization_14_layer_call_fn_3228971234M¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
p
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’  
7__inference_batch_normalization_14_layer_call_fn_322910e1234;¢8
1¢.
(%
inputs’’’’’’’’’00 
p 
Ŗ " ’’’’’’’’’00  
7__inference_batch_normalization_14_layer_call_fn_322923e1234;¢8
1¢.
(%
inputs’’’’’’’’’00 
p
Ŗ " ’’’’’’’’’00 ķ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323102MNOPM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 ķ
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323120MNOPM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 Č
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323138rMNOP;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ "-¢*
# 
0’’’’’’’’’@
 Č
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_323156rMNOP;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ "-¢*
# 
0’’’’’’’’’@
 Å
7__inference_batch_normalization_15_layer_call_fn_323045MNOPM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Å
7__inference_batch_normalization_15_layer_call_fn_323058MNOPM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@ 
7__inference_batch_normalization_15_layer_call_fn_323071eMNOP;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ " ’’’’’’’’’@ 
7__inference_batch_normalization_15_layer_call_fn_323084eMNOP;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ " ’’’’’’’’’@ķ
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323236abcdM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 ķ
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323254abcdM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 Č
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323272rabcd;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ "-¢*
# 
0’’’’’’’’’@
 Č
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_323290rabcd;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ "-¢*
# 
0’’’’’’’’’@
 Å
7__inference_batch_normalization_16_layer_call_fn_323179abcdM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Å
7__inference_batch_normalization_16_layer_call_fn_323192abcdM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@ 
7__inference_batch_normalization_16_layer_call_fn_323205eabcd;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ " ’’’’’’’’’@ 
7__inference_batch_normalization_16_layer_call_fn_323218eabcd;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ " ’’’’’’’’’@š
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323397}~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 š
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323415}~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ė
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323433u}~<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 Ė
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_323451u}~<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 Č
7__inference_batch_normalization_17_layer_call_fn_323340}~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Č
7__inference_batch_normalization_17_layer_call_fn_323353}~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’£
7__inference_batch_normalization_17_layer_call_fn_323366h}~<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’£
7__inference_batch_normalization_17_layer_call_fn_323379h}~<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’ó
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323531N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323549N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ī
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323567x<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 Ī
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_323585x<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 Ė
7__inference_batch_normalization_18_layer_call_fn_323474N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_18_layer_call_fn_323487N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’¦
7__inference_batch_normalization_18_layer_call_fn_323500k<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’¦
7__inference_batch_normalization_18_layer_call_fn_323513k<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’ó
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323665„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323683„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ī
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323701x„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 Ī
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_323719x„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 Ė
7__inference_batch_normalization_19_layer_call_fn_323608„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_19_layer_call_fn_323621„¦§ØN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’¦
7__inference_batch_normalization_19_layer_call_fn_323634k„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’¦
7__inference_batch_normalization_19_layer_call_fn_323647k„¦§Ø<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’¾
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_323832hĒÄĘÅ4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
 ¾
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_323866hĘĒÄÅ4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 
7__inference_batch_normalization_20_layer_call_fn_323799[ĒÄĘÅ4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’
7__inference_batch_normalization_20_layer_call_fn_323812[ĘĒÄÅ4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’§
C__inference_dense_4_layer_call_and_return_conditional_losses_323776`¹ŗ0¢-
&¢#
!
inputs’’’’’’’’’$
Ŗ "&¢#

0’’’’’’’’’
 
(__inference_dense_4_layer_call_fn_323766S¹ŗ0¢-
&¢#
!
inputs’’’’’’’’’$
Ŗ "’’’’’’’’’¦
C__inference_dense_5_layer_call_and_return_conditional_losses_323913_ŠŃ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 ~
(__inference_dense_5_layer_call_fn_323902RŠŃ0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ø
F__inference_dropout_10_layer_call_and_return_conditional_losses_323734n<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ ".¢+
$!
0’’’’’’’’’
 ø
F__inference_dropout_10_layer_call_and_return_conditional_losses_323746n<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ ".¢+
$!
0’’’’’’’’’
 
+__inference_dropout_10_layer_call_fn_323724a<¢9
2¢/
)&
inputs’’’’’’’’’
p 
Ŗ "!’’’’’’’’’
+__inference_dropout_10_layer_call_fn_323729a<¢9
2¢/
)&
inputs’’’’’’’’’
p
Ŗ "!’’’’’’’’’Ø
F__inference_dropout_11_layer_call_and_return_conditional_losses_323881^4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
 Ø
F__inference_dropout_11_layer_call_and_return_conditional_losses_323893^4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 
+__inference_dropout_11_layer_call_fn_323871Q4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’
+__inference_dropout_11_layer_call_fn_323876Q4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’µ
E__inference_dropout_8_layer_call_and_return_conditional_losses_323010l;¢8
1¢.
(%
inputs’’’’’’’’’ 
p 
Ŗ "-¢*
# 
0’’’’’’’’’ 
 µ
E__inference_dropout_8_layer_call_and_return_conditional_losses_323022l;¢8
1¢.
(%
inputs’’’’’’’’’ 
p
Ŗ "-¢*
# 
0’’’’’’’’’ 
 
*__inference_dropout_8_layer_call_fn_323000_;¢8
1¢.
(%
inputs’’’’’’’’’ 
p 
Ŗ " ’’’’’’’’’ 
*__inference_dropout_8_layer_call_fn_323005_;¢8
1¢.
(%
inputs’’’’’’’’’ 
p
Ŗ " ’’’’’’’’’ µ
E__inference_dropout_9_layer_call_and_return_conditional_losses_323305l;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ "-¢*
# 
0’’’’’’’’’@
 µ
E__inference_dropout_9_layer_call_and_return_conditional_losses_323317l;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ "-¢*
# 
0’’’’’’’’’@
 
*__inference_dropout_9_layer_call_fn_323295_;¢8
1¢.
(%
inputs’’’’’’’’’@
p 
Ŗ " ’’’’’’’’’@
*__inference_dropout_9_layer_call_fn_323300_;¢8
1¢.
(%
inputs’’’’’’’’’@
p
Ŗ " ’’’’’’’’’@«
E__inference_flatten_2_layer_call_and_return_conditional_losses_323757b8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’$
 
*__inference_flatten_2_layer_call_fn_323751U8¢5
.¢+
)&
inputs’’’’’’’’’
Ŗ "’’’’’’’’’$ī
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_319554R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_max_pooling2d_6_layer_call_fn_319560R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ī
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_319874R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_max_pooling2d_7_layer_call_fn_319880R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ī
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_320348R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_max_pooling2d_8_layer_call_fn_320354R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’å
O__inference_separable_conv2d_12_layer_call_and_return_conditional_losses_319410%&'I¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 ½
4__inference_separable_conv2d_12_layer_call_fn_319422%&'I¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’ å
O__inference_separable_conv2d_13_layer_call_and_return_conditional_losses_319576ABCI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 ½
4__inference_separable_conv2d_13_layer_call_fn_319588ABCI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@å
O__inference_separable_conv2d_14_layer_call_and_return_conditional_losses_319730UVWI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 ½
4__inference_separable_conv2d_14_layer_call_fn_319742UVWI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@ę
O__inference_separable_conv2d_15_layer_call_and_return_conditional_losses_319896qrsI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ¾
4__inference_separable_conv2d_15_layer_call_fn_319908qrsI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’ź
O__inference_separable_conv2d_16_layer_call_and_return_conditional_losses_320050J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ā
4__inference_separable_conv2d_16_layer_call_fn_320062J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’ź
O__inference_separable_conv2d_17_layer_call_and_return_conditional_losses_320204J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ā
4__inference_separable_conv2d_17_layer_call_fn_320216J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
H__inference_sequential_2_layer_call_and_return_conditional_losses_321968ĘI%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĒÄĘÅŠŃR¢O
H¢E
;8
separable_conv2d_12_input’’’’’’’’’00
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
H__inference_sequential_2_layer_call_and_return_conditional_losses_322102ĘI%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĘĒÄÅŠŃR¢O
H¢E
;8
separable_conv2d_12_input’’’’’’’’’00
p

 
Ŗ "%¢"

0’’’’’’’’’
 
H__inference_sequential_2_layer_call_and_return_conditional_losses_322620³I%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĒÄĘÅŠŃ?¢<
5¢2
(%
inputs’’’’’’’’’00
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
H__inference_sequential_2_layer_call_and_return_conditional_losses_322861³I%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĘĒÄÅŠŃ?¢<
5¢2
(%
inputs’’’’’’’’’00
p

 
Ŗ "%¢"

0’’’’’’’’’
 ė
-__inference_sequential_2_layer_call_fn_320961¹I%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĒÄĘÅŠŃR¢O
H¢E
;8
separable_conv2d_12_input’’’’’’’’’00
p 

 
Ŗ "’’’’’’’’’ė
-__inference_sequential_2_layer_call_fn_321834¹I%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĘĒÄÅŠŃR¢O
H¢E
;8
separable_conv2d_12_input’’’’’’’’’00
p

 
Ŗ "’’’’’’’’’Ų
-__inference_sequential_2_layer_call_fn_322316¦I%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĒÄĘÅŠŃ?¢<
5¢2
(%
inputs’’’’’’’’’00
p 

 
Ŗ "’’’’’’’’’Ų
-__inference_sequential_2_layer_call_fn_322421¦I%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĘĒÄÅŠŃ?¢<
5¢2
(%
inputs’’’’’’’’’00
p

 
Ŗ "’’’’’’’’’
$__inference_signature_wrapper_322211ēI%&'1234ABCMNOPUVWabcdqrs}~„¦§Ø¹ŗĒÄĘÅŠŃg¢d
¢ 
]ŖZ
X
separable_conv2d_12_input;8
separable_conv2d_12_input’’’’’’’’’00"1Ŗ.
,
dense_5!
dense_5’’’’’’’’’