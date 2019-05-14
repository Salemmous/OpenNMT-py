#!/bi/bash
for lang in fr_be sv "fi" nl en
do
	for bpe in "10" "100"
	do
		python3 bpe.py --lang  $lang --num-merges $bpe
	done
done
