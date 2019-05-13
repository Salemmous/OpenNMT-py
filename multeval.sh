#!/bin/bash
MULT=/homeappl/home/deblutst/multeval-0.5.1

cd $MULT

echo "Calculating for each multilingual language"

for lang in en "fi" fr_be nl sv
do
	for bpe in default bpe-10 bpe-100
        do
		[[ $bpe = "default" ]] && train="sign" || train="sign-${bpe}"
		./multeval.sh eval --refs train-set/$train/$bpe/${lang}.txt \
                	--hyps-baseline predictions/multilingual/$bpe/${lang}.txt \
			--latex table-multilingual-${bpe}-${lang}.tex \
                   	--meteor.language $lang
	done
done

echo "Mutl/Mono comparison"
for lang in "fi" nl
do
	for bpe in default bpe-10 bpe-100
        do
		[[ $bpe = "default" ]] && train="sign" || train="sign-${bpe}"
        	./multeval.sh eval --refs train-set/$train/$bpe/${lang}.txt \
        		--hyps-baseline predictions/multilingual/$bpe/${lang}.txt \
        	        --hyps-sys1 predictions/mono/$bpe/${lang} \
			--latex table-mono-mult-${bpe}-${lang}.tex \
                	--meteor.language $lang
	done
done
