#!/bin/bash
rm -rf data/sign
mkdir data/sign
sentences=0
for d in corpora/*/ ; do
    echo "========================================"
    echo "Corpus: $d"
    source "./${d}config"
    mkdir -p data/sign/$lang
    if [[ "$(declare -p gloss)" =~ "declare -a" ]]; then
        arraylength=${#gloss[@]}
        for (( i=0; i<${arraylength}; i++ ));
        do
            newSentences=`python3 corpus_extractor.py --corpus-folder "$d" --glosses "${gloss[$i]}" --trans "${trans[$i]}" --lang $lang --only-number`
            echo "$newSentences new sentences."
            sentences=$sentences+$newSentences
        done
    else
        newSentences=`python3 corpus_extractor.py --corpus-folder "$d" --glosses "$gloss" --trans "$trans" --lang $lang --only-number`
        echo "$newSentences new sentences."
        sentences=$sentences+$newSentences
    fi
    unset gloss
    unset trans
done
echo "========================================"
sentences=`echo "$sentences" | bc`
echo "$sentences sentences."
