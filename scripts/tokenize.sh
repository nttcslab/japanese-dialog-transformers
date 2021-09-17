src=$1
dst=$2
vocab=$3

d2="${vocab##*/}"
#dstrev="${dst}/${vocab%.*}"
dstrev="${dst}/${d2%.*}"

echo $dstrev
mkdir -p $dstrev
cp $vocab $dstrev

#spm_encode --vocabulary --model=$vocab --output_format=piece < $src/train.src > $dstrev/train.src
spm_encode --model=$vocab --output_format=piece < $src/train.src > $dstrev/train.src
spm_encode --model=$vocab --output_format=piece < $src/train.dst > $dstrev/train.dst
spm_encode --model=$vocab --output_format=piece < $src/valid.src > $dstrev/valid.src
spm_encode --model=$vocab --output_format=piece < $src/valid.dst > $dstrev/valid.dst
spm_encode --model=$vocab --output_format=piece < $src/test.src > $dstrev/test.src
spm_encode --model=$vocab --output_format=piece < $src/test.dst> $dstrev/test.dst
